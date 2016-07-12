
import random, pickle, os.path, csv, re
import datetime
from collections import defaultdict
from operator import itemgetter

from flask import Flask, render_template
from flask import request, redirect
from flask_socketio import SocketIO
from flask_socketio import join_room, leave_room, emit

from proc_chat import *

application = app = Flask(__name__)

socketio = SocketIO(app)

#-------------------------------------------------------------------------
questions = [
	"<b>HDP:</b> Hello, nice to meet you! May I get your full name? (No accents please)",
	"<b>HDP:</b> Great, would you mind telling us your gender? Male, Female or Neither",
	"<b>HDP:</b> Thanks, can we also get your birthdate? (US Format)",
	"<b>HDP:</b> Got it, how far would you be willing to travel for a clinical trial? (in miles)",
	]

PRE_COND = "<b>HDP:</b> Thanks! We currently have <mark>~400 clinical trials</mark> in our database. "+\
"To help narrow down the trials, we'd like to ask you a few questions.<br><br>"
COND_QUESTION = PRE_COND+"<b>HDP:</b> Are you searching for condition <mark><b>XX</b></mark>? (yes or no)"
INTER_QUESTION = "<b>HDP:</b> Is your preferred intervention type <mark><b>XX</b></mark>? (yes or no)"
KEYWORD_QUESTION = "<b>HDP:</b> Are you currently using <mark><b>XX</b></mark> (an exclusion criteria)? (yes or no)"

#-------------------------------------------------------------------------
def get_greedy(G, node):
	x = G[node]
	sorted_x = sorted(x.items(), key=itemgetter(1), reverse=True)
	return sorted_x



#-------------------------------------------------------------------------
# all trials' ID
all_trials = open('trials.csv').read().split('\n')
all_trials = all_trials[1:]
all_trials = [t.strip() for t in all_trials]
print 'Number of trials:', len(all_trials)

# other pre-processed info
G_cond_inter_keyword = pickle.load(open('G_cond_inter_keyword.pickle','rb'))
cond2trials = pickle.load(open('cond2trials.pickle','rb'))
alltype2tid2q = pickle.load(open('alltype2tid2q.pickle','rb'))
excl_w2trials = pickle.load(open('excl_w2trials.pickle','rb'))


#-------------------------------------------------------------------------

# globals
cond_tuples = get_greedy(G_cond_inter_keyword, 'system')
cond_question_list = [c for c, v in cond_tuples]


#-------------------------------------------------------------------------

results = {}

@socketio.on('trial chat send', namespace='/chat')
def on_message(message):
	# time.sleep(1)
	print '----> current message:', int(message['qid'])
	if int(message['qid']) == -1:
		emit('trial chat response', 
			{'message': '<br><br>' + 'please click start/restart chat', 'qid': -1 }, 
			room=message['room'], namesapce='/chat')
		return
	if int(message['qid']) > 7:
		emit('trial chat response', 
			{'message': '<br><br>' + 'please click start/restart chat', 'qid': -1 }, 
			room=message['room'], namesapce='/chat')
		return
	
	qid, ret_str, success = main(int(message['qid']), message['message'], message['room'])

	if ret_str: ret_str=ret_str+'<br>'
	ret_str = re.sub('\n','<br>',ret_str)

	if not success:
		pass
		# emit notify specialist
		qid = 10

	elif qid == 4:
		# next question is condition
		cond = cond_question_list[results[message['room']]['cond_idx']]
		response = re.sub('XX', cond, COND_QUESTION)

	elif qid == 5:
		# show remaining trials after selecting condition
		c = cond_question_list[results[message['room']]['cond_idx']-1]
		remain_trials = len(cond2trials[c]) #cond_map[c]
		perc = float(remain_trials) / len(all_trials)
		response = "<b>HDP:</b> You've selected "+c+\
			", which narrows down to <mark>"+str(remain_trials)+\
			"</mark> trials ("+"~%.2f"%(perc*100)+"% of all trials)<br><br>"
		# next question is intervention type
		types_list = results[message['room']]['types_list']
		inter = types_list[results[message['room']]['inter_idx']]
		response += re.sub('XX', inter, INTER_QUESTION)

	elif qid == 6:
		# show remaining trials after condition and intervention type
		c = cond_question_list[results[message['room']]['cond_idx']-1]
		types_list = results[message['room']]['types_list']
		t = types_list[results[message['room']]['inter_idx']-1]
		c_t = c+'-'+t
		remain_trials = len(alltype2tid2q[c_t]) 
		perc = float(remain_trials) / len(all_trials)
		response = "<b>HDP:</b> You've selected "+c_t+\
			", which narrows down to <mark>"+str(remain_trials)+\
			"</mark> trials ("+"~%.2f"%perc+" of all trials)<br><br>"
		# next question is the keyword
		words_list = results[message['room']]['keywords_list']
		k = words_list[results[message['room']]['keyword_idx']]
		response += re.sub('XX', k, KEYWORD_QUESTION)

	elif qid == 7:
		# show remaining trials after condition, intervention, and keyword
		c = cond_question_list[results[message['room']]['cond_idx']-1]
		types_list = results[message['room']]['types_list']
		t = types_list[results[message['room']]['inter_idx']-1]

		words_list = results[message['room']]['keywords_list']
		w = words_list[results[message['room']]['keyword_idx']-1]

		c_t = c+'-'+t

		all_avoid_trials = set(alltype2tid2q[c_t].keys()) & set(excl_w2trials[w])
		remain_trials = set(alltype2tid2q[c_t].keys())-set(all_avoid_trials)
		remain_trials = list(remain_trials)
		perc = float(len(remain_trials)) / len(all_trials)

		response = "<b>HDP:</b> You've selected "+c_t+'-'+w+\
			", which narrows down to <mark>"+str(len(remain_trials))+\
			"</mark> trials ("+"~%.2f"%(perc*100)+"% of all trials)<br><br>"

		# get all responses
		response += get_all_info(message['room'])
		#### get a partial list of remaining trials and links
		tids = remain_trials[:6] 
		ret_str += '<br>The first few recommended trials are as follow:<br>'
		cnt = 0
		for tid in tids:
			url = 'https://clinicaltrials.gov/ct2/show/'+tid
			ret_str += '<a href="'+url+'" target="_blank">'+tid+\
						'</a>&nbsp;&nbsp'
			cnt+=1
			if cnt%2==0: ret_str+='<br>'
		print '----> response:', type(ret_str) #, response

	else:
		try:
			response = questions[qid]
		except:
			response = qid
			qid = 10

	emit('trial chat response', {'message': '<br><br>' + response, 'qid': qid }, \
		room=message['room'], namesapce='/chat')
	emit('trial chat response userinfo', {'message': ret_str}, \
		room=message['room'], namespace='/chat')


@socketio.on('start', namespace='/chat')
def on_message(message):
	results[message['room']] = {
		'name': None,
		'gender': None,
		'birth_year': None,
		'birth_month': None,
		'birth_day': None,
		'age': None,
		'dist': None,
		'treatments': None,
		'condition': None,
		'intervention': None,
		'keyword': None,
		'cond_idx': 0,
		'inter_idx': 0,
		'keyword_idx': 0,
		'types_list': None,
		'keywords_list': None,
	}
	emit('trial chat response', {'message': questions[0], 'qid': 0 }, \
		room=message['room'], namesapce='/chat')

@socketio.on('join', namespace='/chat')
def on_join(room):
	join_room(room)

@socketio.on('leave', namespace='/chat')
def on_leave(room):
  leave_room(room)



def get_trials2(treatments):
	return_str = ''
	exclusion = []
	for t in treatments:
		t = t.strip()
		# collect all exclusion trials
		if t in excl_w2trials:
			for tid in excl_w2trials[t]:
				if tid not in exclusion:
					exclusion.append(tid)

	selected_trials = list(set(all_trials)-set(exclusion))
	#print selected_trials
	return_str += "<br>Number of selected trials: <b>"+str(len(selected_trials))+"</b> "
	return_str += "total: "+str(len(all_trials))+" (excluded: "+str(len(exclusion))+")<br>"
	print return_str
	return selected_trials, return_str


def main_treatment(rsp):
	#print rsp
	treatments = get_treatments(rsp)
	#print treatments
	return_str ='Treatments: '+str(treatments)+'<br>'	
	selected_trials, str1 = get_trials2(treatments)
	return_str+=str1
	return_str+='First few trial IDs: '
	for tid in selected_trials[:5]:
		#print tid, tid in trials_criteria2
		url = trials_criteria2[tid]['url']
		url = re.sub("\?resultsxml=true","",url)
		return_str+="<a href=\""+url+"\" target=\"_blank\">"+tid+"</a> "
		#print return_str
	return_str+="...<br>"
	return return_str


def main(qid, rsp, room):
	global cond_idx, inter_idx, keyword_idx
	global name, gender, birth_year, birth_month, birth_day
	global age, dist, treatments, condition, intervention, keyword

	#logname = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+'_'+str(room)+'.log'
	logname = str(room)+'.log'
	with open(os.path.join('chatlog',logname), 'ab') as f:
		f.write(rsp+'\n')

	print '----> main room:', type(room), room, 'qid:', qid, ', rsp:', type(rsp), rsp

	return_str = ''
	treatment_str = ''

	if qid==0: 
		#name = get_name(rsp)
		results[room]['name'] = get_name(rsp)
		return_str='<b>Name:</b> '+results[room]['name']
		print '---->', return_str

	elif qid==1:
		#gender = get_gender(rsp)
		results[room]['gender'] = get_gender(rsp)
		return_str='<b>Gender:</b> '+results[room]['gender']
		print '---->', return_str

	elif qid==2:
		results[room]['birth_year'], results[room]['birth_month'], \
			results[room]['birth_day'] = get_birthdate(rsp)
		if results[room]['birth_year'] and results[room]['birth_month'] and \
			results[room]['birth_day']:
			results[room]['age'] = calculate_age(results[room]['birth_year'], 
				results[room]['birth_month'], results[room]['birth_day'])
		#birth_year, birth_month, birth_day = get_birthdate(rsp)
		#if birth_year and birth_month and birth_day:
		#	age = calculate_age(birth_year, birth_month, birth_day)
		return_str='<b>Birth m/d/y:</b> '+ str(results[room]['birth_month'])+ \
				'/'+str(results[room]['birth_day'])+\
				'/'+str(results[room]['birth_year'])+'\n'+ \
				'<b>Age:</b> '+str(results[room]['age'])
		print '---->', return_str

	elif qid==3:
		#dist = get_distance(rsp)
		results[room]['dist'] = get_distance(rsp)
		return_str = '<b>Dist:</b> '+str(results[room]['dist'])
		print '---->', return_str

	elif qid==4:
		#treatments = get_treatments(rsp)
		#treatment_str = main_treatment(rsp)
		#results[room]['treatments'] = get_treatments(rsp)
		# get condition
		yesno = rsp.strip().lower()
		#condition = cond_question_list[cond_idx]
		results[room]['condition'] = cond_question_list[results[room]['cond_idx']]
		return_str = '<b>Condition:</b> '+results[room]['condition']
		print '---->', return_str, yesno
		results[room]['cond_idx']+=1

		if yesno[0]!='y':
			# decrement qid to repeat this q
			qid-=1
			return_str = ''
		else:
			# got condition; get types
			types_tuple = get_greedy(G_cond_inter_keyword, results[room]['condition'])
			types_list = [t for t, v in types_tuple]
			results[room]['types_list'] = types_list

	elif qid==5:
		# intervention
		yesno = rsp.strip().lower()
		#intervention = inter_question_list[inter_idx]
		idx = results[room]['inter_idx']
		results[room]['intervention'] = results[room]['types_list'][idx] 
		return_str = '<b>Intervention type:</b> '+results[room]['intervention']
		print '---->', return_str, ':', yesno
		results[room]['inter_idx']+=1

		if yesno[0]!='y':
			# decrement qid to repeat this q
			qid-=1
			return_str = ''
		else:
			# got intervention type; get words
			c = results[room]['condition']
			t = results[room]['intervention']
			c_t = c+'-'+t
			keywords_tuple = get_greedy(G_cond_inter_keyword, c_t)
			keywords_list = [k for k, v in keywords_tuple]
			results[room]['keywords_list'] = keywords_list


	elif qid==6:
		# keyword
		yesno = rsp.strip().lower()
		#keyword = keyword_question_list[keyword_idx]
		idx = results[room]['keyword_idx']
		results[room]['keyword'] = results[room]['keywords_list'][idx] 
		return_str = '<b>Keyword:</b> '+results[room]['keyword']
		print '---->', return_str, yesno
		results[room]['keyword_idx']+=1

		if yesno[0]!='y':
			qid-=1
			return_str = ''

	return qid + 1, return_str, True



def get_all_info(room):
	global name, gender, birth_year, birth_month, birth_day
	global age, dist, treatments, condition, intervention, keyword

	# if it's the last question, then save data and retrieve trials
	return_str = ''
	return_str+= "<b>HDP:</b> Got it. An initial list of "+\
			"trial options are shown on the right. <br>"

	# insert user info to db
	"""
	# new id = get the last row then add 1 to it
	query_rsp = pd.read_sql_query("select id from "+PEOPLE_DB+\
				" order by id desc limit 1;", con)
	id = query_rsp.iloc[0][0]+1
	sql_query2 = insert_table_query(id, name, gender, birth_year, birth_month, 
					birth_day, age, dist, treatments)
	#print sql_query2
	cur = con.cursor()
	cur.execute(sql_query2)
	con.commit()	
	return_str+='<br>*** Patient added to DB ('+PEOPLE_DB+') with id:'+str(id)+'***<br><br>'
	"""

	return_str+='<br>*** Your info is added to the local chatlog: '+str(room)+'***<br><br>'


	# rest variables
	results[room] = {
		'name': None,
		'gender': None,
		'birth_year': None,
		'birth_month': None,
		'birth_day': None,
		'age': None,
		'dist': None,
		'treatments': None,
		'condition': None,
		'intervention': None,
		'keyword': None,
		'cond_idx': 0,
		'inter_idx': 0,
		'keyword_idx': 0,
		'types_list': None,
		'keywords_list': None,
	}

	#print '----> return_str:', return_str
	return return_str


@app.route('/', methods=['GET','POST'])
def index():
	return render_template('index.html')

if __name__ == '__main__':
	socketio.run(app) #, host='0.0.0.0')
