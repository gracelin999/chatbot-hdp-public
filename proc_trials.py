import re, urllib2, time, csv
import xmltodict, codecs, os, pickle
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from spacy.en import English
import string, math, random

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
#from nltk.corpus import stopwords

from operator import itemgetter
import copy

from utils import int_to_roman
from fuzzywuzzy import fuzz

#----------------------------------------------------------------------

en_parser = English()

#----------------------------------------------------------------------

incl_w2trials = None
excl_w2trials = None
trials_criteria = None
trials_criteria2 = None

if os.path.isfile('incl_w2trials.pickle'):
	incl_w2trials = pickle.load(open('incl_w2trials.pickle'))
	# make sure there's no duplicating trials for a word
	for w in incl_w2trials:
		if not w: del incl_w2trials[w]
		incl_w2trials[w]=list(set(incl_w2trials[w]))

if os.path.isfile('excl_w2trials.pickle'):
	excl_w2trials = pickle.load(open('excl_w2trials.pickle'))
	# make sure there's no duplicating trials for a word
	for w in excl_w2trials:
		excl_w2trials[w]=list(set(excl_w2trials[w]))

if os.path.isfile('trials_criteria.pickle'):
	trials_criteria = pickle.load(open('trials_criteria.pickle'))
else:
	trials_criteria = defaultdict(dict)
	csvreader = csv.reader(open('trials_criteria.csv'))
	next(csvreader) # skip header
	for tid, url, condition, criteria, intervention in csvreader:
		tid = tid.strip()
		if not tid: continue
		trials_criteria[tid]['condition']=condition.strip()
		trials_criteria[tid]['intervention']=intervention.strip()
	pickle.dump(trials_criteria, open('trials_criteria.pickle','wb'))


if os.path.isfile('trials_criteria2.pickle'):
	trials_criteria2 = pickle.load(open('trials_criteria2.pickle'))
else:
	trials_criteria2 = defaultdict(dict)
	csvreader = csv.reader(open('trials_criteria2.csv','rb'))
	next(csvreader) # skip header
	for tid, url, incl, excl, incl_w, excl_w in csvreader:
		tid = tid.strip()
		if not tid: continue
		trials_criteria2[tid]['url']=url.strip()
		trials_criteria2[tid]['incl']=incl.strip()
		trials_criteria2[tid]['excl']=excl.strip()
		trials_criteria2[tid]['incl_w']=incl_w.strip()
		trials_criteria2[tid]['excl_w']=excl_w.strip()
	pickle.dump(trials_criteria2, open('trials_criteria2.pickle','wb'))

if not incl_w2trials or not excl_w2trials:
	# create an index (with words) to trial IDs
	incl_w2trials = defaultdict(list)
	excl_w2trials = defaultdict(list)

	for tid in trials_criteria2:
		url = trials_criteria2[tid]['url']
		incl = trials_criteria2[tid]['incl']
		excl = trials_criteria2[tid]['excl']
		incl_w = trials_criteria2[tid]['incl_w']
		excl_w = trials_criteria2[tid]['excl_w']
		for w in incl_w.split(','):
			w = w.lower().strip()
			#print w, tid
			incl_w2trials[w].append(tid)
		for w in excl_w.split(','):
			w = w.lower().strip()
			excl_w2trials[w].append(tid)
	# save them
	pickle.dump(incl_w2trials, open('incl_w2trials.pickle','wb'))
	pickle.dump(excl_w2trials, open('excl_w2trials.pickle','wb'))

# clean up
if '' in incl_w2trials: del incl_w2trials['']
if '' in excl_w2trials: del excl_w2trials['']
if '' in trials_criteria: del trials_criteria['']
if '' in trials_criteria2: del trials_criteria2['']

#----------------------------------------------------------------------

def get_xml_fields(entry):
	intervention_type = entry['intervention_type']
	intervention_name = entry['intervention_name']

	if 'description' in entry:
		intervention_desc = entry['description'] 
	else:
		intervention_desc = ''

	#print data['intervention']
	if 'arm_group_label' in entry:
		intervention_arm_group_label = entry['arm_group_label']
		if type(intervention_arm_group_label)==list:
			intervention_arm_group_label = ','.join(intervention_arm_group_label)
	else:
		intervention_arm_group_label = ''
	
	if 'other_name' in entry:
		intervention_other_name = entry['other_name']
		if type(intervention_other_name)==list:
			intervention_other_name = ','.join(intervention_other_name)
	else:
		intervention_other_name = ''

	all = '|'.join([intervention_type, intervention_name, \
		intervention_desc, intervention_arm_group_label, \
		intervention_other_name])
	return all

#----------------------------------------------------------------------

def get_from_web():
	trials = open('trials.csv').read().split('\n')
	trials = trials[1:]
	print 'Number of trials:', len(trials)

	base_url = "https://clinicaltrials.gov/ct2/show/XX?resultsxml=true"

	csvwriter = csv.writer(open('trials_criteria.csv','wb'))
	csvwriter.writerow(['trial_id', 'url', 'condition', 'criteria', \
		'intervention_type_name_desc_label_other'])

	trials_criteria = {}

	trials_wrote = 0

	for trial in trials:
		trial=trial.strip()

		url = re.sub('XX',trial,base_url)
		print
		print 'processing trial:', trial
		print url

		file = urllib2.urlopen(url)
		data = file.read()
		file.close()

		data_raw = xmltodict.parse(data)
		data = data_raw['clinical_study']
		condition = data['condition']
		if type(condition)==list:
			condition = '|'.join(condition)
		criteria = data['eligibility']['criteria']['textblock']

		intervention_all = []
		if 'intervention' in data:
			#if trial=='NCT02093663': print type(data['intervention'])
			if type(data['intervention'])==list:
				for entry in data['intervention']:
					all = get_xml_fields(entry)
					intervention_all.append(all)
			else:
				all = get_xml_fields(data['intervention'])
				intervention_all.append(all)
		
		intervention_all_str = '\n\n'.join(intervention_all)

		#print criteria.encode('utf-8')
		if trial == 'NCT02412085':
			print type(condition), type(criteria), type(intervention_all_str)

		csvwriter.writerow([trial,url,
			condition.encode('utf-8'), 
			criteria.encode('utf-8'),
			intervention_all_str.encode('utf-8')])
		trials_wrote+=1

		time.sleep(0.1)

	print 'Number of trials gathered:', trials_wrote
	

#----------------------------------------------------------------------

def unicode_csv_reader(utf8_data, dialect=csv.excel, **kwargs):
	csv_reader = csv.reader(utf8_data, dialect=dialect, **kwargs)
	for row in csv_reader:
		yield [unicode(cell, 'utf-8') for cell in row]

#----------------------------------------------------------------------

def get_drug_words():
	csvwriter = csv.writer(open('trials_criteria2.csv','wb'))
	csvwriter.writerow(['trial_id', 'url', 'inclusion','exclusion',\
		'inclusion_words','exclusion_words'])

	from proc_chat import get_treatments

	#csvreader = unicode_csv_reader(open('trials_criteria.csv'))
	csvreader = csv.reader(open('trials_criteria.csv'))
	next(csvreader) # skip header
	for trial_id, url, condition, criteria, intervention in csvreader:
		#if trial_id != 'NCT02044952': continue

		print 'Processing', trial_id
		arr = criteria.split('Exclusion Criteria:')
		if len(arr)==1:
			exclusion = ''
		else:
			exclusion = arr[1].strip()
		inclusion = re.sub('Inclusion Criteria:','',arr[0]).strip()
		#
		#inclusion = fix_unicode(inclusion)
		#exclusion = fix_unicode(exclusion)
		#print 'inclusion:', inclusion
		# get words
		inclusion_words = []
		exclusion_words = []
		if inclusion:
			inclusion_words = get_treatments(inclusion)
		if exclusion:
			exclusion_words = get_treatments(exclusion)

		# write to csv
		csvwriter.writerow([trial_id,url,inclusion,exclusion, \
					','.join(inclusion_words), 
					','.join(exclusion_words)])

#----------------------------------------------------------------------

def plot_wc(wc):
	import matplotlib.pyplot as plt 
	from wordcloud import WordCloud
	wordcloud = WordCloud().generate(' '.join(wc.keys()))
	plt.imshow(wordcloud)
	plt.show()

#----------------------------------------------------------------------

def tokenize_text(content):
	# A custom stoplist
	STOPLIST = list(ENGLISH_STOP_WORDS)
	# List of symbols we don't care about
	SYMBOLS = " ".join(string.punctuation).split(" ") + ["-----", "---", "...", "'ve"]

	parsed_data = en_parser(content.decode('utf-8'))

	# lemmatize
	lemmas = []
	for tok in parsed_data:
		if tok.is_stop: continue
		if tok.lemma_ == "-PRON-": continue
		lemmas.append(tok.lemma_.lower().strip())
		#else:
		#	lemmas.append(tok.lower_)

	tokens = lemmas

	# stoplist the tokens
	tokens = [tok for tok in tokens if tok not in STOPLIST]

	# stoplist symbols
	tokens = [tok for tok in tokens if tok not in SYMBOLS]

	# remove large strings of whitespace
	while "" in tokens:
		tokens.remove("")
	while " " in tokens:
		tokens.remove(" ")
	while "\n" in tokens:
		tokens.remove("\n")
	while "\n\n" in tokens:
		tokens.remove("\n\n")

	return tokens

#----------------------------------------------------------------------

def get_words_cutoff(wc, cutoff):
	# look at all the words that make up certain % of total word count
	words = []
	counts = []
	#ratios = []
	ptotal = 0.0

	if sum(wc.values())!=0:
		for w, c in wc.most_common():
			words.append(w)
			counts.append(c)
			# proportion
			p = float(c)/sum(wc.values())
			#ratios.append(p)
			ptotal+=p
			if ptotal>cutoff:
				break

	return ptotal, words

#----------------------------------------------------------------------

def get_tfidf(corpus):
	tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, \
			stop_words = 'english')
	tfidf_matrix =  tf.fit_transform(corpus)
	feature_names = tf.get_feature_names()
	print '[tfidf] num features:', len(feature_names)
	print '[tfidf] feat names:', feature_names[:3], '...'
	return tfidf_matrix, feature_names


#----------------------------------------------------------------------

def get_entry(entry_orig, dense_entry, feature_names, q2w, w2q):
	# look at this entry/trial only
	raw1_arr = entry_orig.split('\n')
	raw1_arr = [t.lower().strip() for t in raw1_arr if t]
	#print '[corpus]', raw1_arr

	entry1 = dense_entry.tolist()[0]
	#print '[tfidf] num feats:',len(entry1)
	phrase_scores = \
		[pair for pair in zip(range(0, len(entry1)), entry1) if pair[1] > 0]
	#print '[tfidf] num feats for 1st entry:', len(phrase_scores), \
	#		len(phrase_scores)/len(entry1), \
	#		phrase_scores[:3], '...'

	# sort phrases by score in DECREASING order to find the LEAST interesting
	# ones first; this shows what's most frequently occurred through all

	words_local = []
	q_local = []

	phrase_scores_sorted = sorted(phrase_scores, key=lambda t: t[1])
	for widx, score in phrase_scores_sorted[:10]:
		word = feature_names[widx]
		#txt = '{0: <20} {1}'.format(word, score)
		#found = False
		for raw in raw1_arr:
			#print type(word), word
			#print type(raw), raw
			if word in raw:
				#txt += '\t[corpus]: '+raw+'\n'
				q2w[raw].append(word)
				w2q[word].append(raw)
				if word not in words_local:
					words_local.append(word)
				if raw not in q_local:
					q_local.append(raw)
				#found = True
		#if found:
		#	print txt
	#print '...'

	#raw_counter_reverse = sorted(raw_counter.items(), key=itemgetter(1), reverse=False)
	#print raw_counter
	#print len(w2q.keys()), w2q.keys()[:5], '...'
	#for w in w2q:
	#	print w, len(w2q[w])

	#return raw_counter_reverse
	return words_local, q_local, phrase_scores_sorted

#----------------------------------------------------------------------

def clean_clean(tid, content, content1, start=None):
	found_list = False
	found_roman_numeral = False

	int_arr = re.findall('(\d+)\. ',content)
	#if '1. ' in content or '2. ' in content:
	if int_arr:
		found_list = True
	elif 'i. ' in content or 'ii. ' in content:
		found_list = True
		found_roman_numeral = True
		print 'found roman numeral:', tid

	if found_list:
		content_new = re.sub('^- ','', content1[0])
		criteria_new = []
		# then start splitting the rest
		n_start = start+1 if start else 2
		n = n_start
		while True:
			n_str = int_to_roman(n).lower() if found_roman_numeral else str(n)
			arr = content_new.split(n_str+'. ',1)
			if n==n_start:
				if found_roman_numeral:
					match = int_to_roman(n-1).lower()+'. '
				else:
					match = str(n-1)+'. '
				#if tid=='NCT02289417': 
				#	print '---->', arr[0], match in arr[0], match
				if match in arr[0]:
					# check n == 1 in arr[0]
					arr22 = arr[0].split(match,1)
					#print tid, arr22
					# is there something before '1. '?
					if arr22[0].strip():
						# if so, then add that first
						criteria_new.append(arr22[0])
						#print arr[0], arr2
					criteria_new.append(match+arr22[1])
				else:
					criteria_new.append(arr[0])
			else:
				criteria_new.append(arr[0])
			if len(arr)==1:
				break
			content_new = n_str+'. '+arr[1].strip()
			n+=1
			if n_str+'.' not in content_new:
				criteria_new.append(content_new)
				break

		# store previous
		content1_before = copy.copy(content1)
		# write new
		content1 = ['- '+re.sub('\s+',' ',s.lower().strip()) for s in criteria_new if s]

		# sanity check before and after
		before_str = re.sub('^- ','', content1_before[0]).strip()
		after_str = ''
		for s in content1:
			after_str+=re.sub('^- ','',s)+' '
		after_str = after_str.strip()

		if before_str!=after_str:
			print 'WARN. Before and after DO NOT match:', tid
			print '[before]:', len(content1_before) #, content1_before[:2], '...'
			print '[after]:', len(content1) #, content1[:2], '...'

	return content1

#----------------------------------------------------------------------

def criteria_cleanup(tid, content):
	content1 = content.split('-  ')
	content1 = ['- '+re.sub('\s+',' ',s.lower().strip()) for s in content1 if s]
	

	# only has 1 criteria;
	# see if it's listed with 1, 2, 3, ... instead of bullet points
	if len(content1)==1:
		content1 = clean_clean(tid, content, content1)
	else:
		content_new = []
		for c in content1:
			start = None
			arr = re.findall('(\d+)\. ',c)
			if arr: 
				start=int(arr[0])
				content_new += clean_clean(tid, c, [c], start=start)
			else:
				content_new += [c]
		content1 = content_new

	if 'subject with gastro-intestinal tract' in content.lower():
		print '=====>', tid

	"""
	if 'history of colonic or rectal surgery' in content.lower() or \
		'presence of clinically significant cardiovascular' in content.lower():

		print '-----------> Additional process for', tid, len(content1)
		content_new = []
		for c in content1:
			arr = re.findall('(\d+)\. ',c)
			if arr: 
				start=int(arr[0])
				content_new += clean_clean(tid, c, [c], start=start)
			else:
				content_new += [c]
		content1 = content_new
		for i, c in enumerate(content1):
			print '[',i,']', c
	"""
	content1_str = '\n'.join(content1)
	return content1, content1_str

#----------------------------------------------------------------------

def get_top_words(G_drug, G_tfidf, \
		trials_criteria, trials_criteria2, incl_w2trials, excl_w2trials):

	wc = Counter()
	wc_content = Counter()
	wc_content2trials = defaultdict(list)
	corpus = []

	print 'num of trials_criteria:', len(trials_criteria)
	print 'num of trials_criteria2:', len(trials_criteria2)

	if len(trials_criteria)!=len(trials_criteria2):
		print 'ERROR. Number of trials do not match:'
		print 'Trials criteria:', trials_criteria.keys()
		print 'Trials criteria2:', trials_criteria2.keys()

	all_trials = trials_criteria2.keys()
	tid2q = {}

	for tid in trials_criteria2:
		# look at words
		excl_w = trials_criteria2[tid]['excl_w']
		for w in excl_w.split(','):
			w = w.strip()
			if not w: continue
			# TODO: skip certain words for now due to parsing errors
			#if w in ['protection','start','solution','regimen','care', 'body',
			#	'release','screen']: 
			#	continue
			wc[w]+=1

		# look at words from content
		content = trials_criteria2[tid]['excl']
		tokens = tokenize_text(content)

		for t in tokens:
			wc_content[t]+=1
			if tid not in wc_content2trials[t]:
				wc_content2trials[t].append(tid)

		# store as corpus; clean up a bit first
		#content1 = content.split('-  ')
		#content1 = ['- '+re.sub('\s+',' ',s.lower().strip()) for s in content1 if s]
		#content1_str = '\n'.join(content1)
		content1, content1_str = criteria_cleanup(tid, content)
		#if len(content1)<=2:
		#	print tid, len(content1)
		if content1_str:
			corpus.append(content1_str.decode('utf-8'))
		if content1:
			tid2q[tid]=content1

	q2w = defaultdict(list)
	w2q = defaultdict(list)
	q2ntrials = defaultdict(list)
	w2ntrials = defaultdict(list)

	if not len(corpus):
		print 'NOTE. Empty corpus. Skip looking for words.'

	else:
		# find corresponding number of trials for each word
		print '[drug] top excl words and % trials (total: ', len(corpus),')'
		trial_counts = Counter()
		for w in excl_w2trials:
			w = w.strip()
			if not w: continue
			if w in trial_counts: continue
			num_trials = len(excl_w2trials[w])
			#if len(trials_criteria)==1:
			#	print w, excl_w2trials[w]
			trial_counts[w]= num_trials
		for w, c in trial_counts.most_common(10):
			score = float(c)/len(corpus)
			print '\t{0: <20} {1}'.format(w.encode('utf-8'), score*100) 
			G_drug[w]=score
		# look at the cutoff
		p, words = get_words_cutoff(trial_counts, 0.6)
		print '[drug] To get at least',p*100,'% of all trials we need at least', len(words), 'words'
		

		#---------------------------------------------
		print '[drug] top excl words and % words (total:', sum(wc.values()),')'
		for w, c in wc.most_common(10):
			score =float(c)/sum(wc.values())*100
			print '\t{0: <20} {1}'.format(w.encode('utf-8'), score)
		# look at the cutoff
		p, words = get_words_cutoff(wc, 0.6)
		print '[drug] To get at least',p*100,'% of all words we need at least', len(words), 'words'


		print '[content] top excl words and % trials (total:', len(corpus),')'
		wc_content2trials_sorted = sorted(wc_content2trials, \
			key=lambda e: len(wc_content2trials[e]), reverse=True)
		for w in wc_content2trials_sorted[:10]:
			c = len(wc_content2trials[w])
			score =float(c)/len(corpus)
			print '\t{0: <20} {1}'.format(w.encode('utf-8'), score*100) 
			G_tfidf[w]=score

		#---------------------------------------------
		print '[content] top excl words and % words (total:', sum(wc_content.values()),')'
		for w, c in wc_content.most_common(10):
			score = float(c)/sum(wc_content.values())
			print '\t{0: <20} {1}'.format(w.encode('utf-8'), score*100)
			#G_tfidf[w]=score
		# look at the cutoff
		p_content, words_content = get_words_cutoff(wc_content, 0.6)
		print '[content] To get at least',p_content*100,'% of all words we need at least', len(words_content), 'words'


		# plot wordcloud
		#plot_wc(wc)

		"""
		pos = np.arange(len(ratios)-1,-1,-1)+0.5
		plt.barh(pos,ratios, align='center')
		plt.yticks(pos, words)
		plt.title('Top Exclusion Words From Trials')
		plt.xlabel('Percent of all exclusion words from trials')
		plt.ylim([0,len(ratios)])
		plt.grid()
		plt.savefig('top_words2.png',bbox_inches='tight')
		"""
		#plt.show()


		# replace corpus with criteria
		"""
		corpus = []
		corpus = ['Known GI related symptoms complaints or GI diseases.',
			'Swallowing disorders',
			'Cancer or other life threatening diseases or conditions',
	        'Pregnancy or breast-feeding',
	        'Previous abdominal surgery',
	        'Abdominal diameter >140cm?',
	        'Drug abuse or alcoholism',
	        'Irregular bowel movements',
	        'Known cardiovascular or pulmonary diseases',
	        'Participation in any clinical study within the last 30 days',
	        'Cardiac pacemaker or infusion pump or any other implanted or portable',
	        'electro-mechanical medical device.',
	        'Medication affecting GI motility',
			'MRI within the next four weeks'
			]
		"""
		print

		tfidf_matrix, feature_names = get_tfidf(corpus)
		dense = tfidf_matrix.todense()
		#print '[tfidf]',dense[:3],'...'
		# very sparse data; will see mostly 0's

		#criteria_counter = Counter()
		#q2w = defaultdict(list)
		#w2q = defaultdict(list)

		#q2ntrials = defaultdict(list)
		#w2ntrials = defaultdict(list)

		# go through each trial
		for i in range(len(corpus)):
			words_local, q_local, phrase_scores_sorted = \
				get_entry(corpus[i], dense[i], feature_names, q2w, w2q)
			for w in words_local:
				w2ntrials[w].append(i)
				w2ntrials[w]=list(set(w2ntrials[w]))
			for q in q_local:
				q2ntrials[q].append(i)
				q2ntrials[q]=list(set(q2ntrials[q]))

		print '[tfidf] w2ntrials top words and % trials (total:', len(corpus),')'
		w2ntrials_sorted = sorted(w2ntrials, key=lambda e: len(w2ntrials[e]), reverse=True)
		for w in w2ntrials_sorted[:20]:
			score = float(len(w2ntrials[w]))/len(corpus)
			#print type(w)
			print '\t{0: <20} {1}'.format(w.encode('utf-8'), score*100)
			#G_tfidf[w]=score

		#print '[tfidf] q2ntrials:', len(corpus), q2ntrials.most_common(3), '...'


		"""
		# not the most helpful here
		print '[tfidf] q2w:', len(q2w)
		q2w_reverse = sorted(q2w, key=lambda e: len(q2w[e]), reverse=True)
		for q in q2w_reverse[:3]:
			print q, len(q2w[q])
			#print '\t',q2w[q]
		print '...'
		for q in q2w_reverse[-3:]:
			print q, len(q2w[q])
		"""

		total_q = sum([len(w2q[w]) for w in w2q])
		print '[tfidf] w2q',len(w2q),'( total q:', total_q,')' #, w2q_reverse[:5], '...'

		w2q_reverse = sorted(w2q, key=lambda e: len(w2q[e]), reverse=True)
		for w in w2q_reverse[:10]:
			#if len(w2q[w])<=1: continue
			score = len(w2q[w])/float(total_q)*100 #, w2q[w]
			print '\t{0: <20} {1}'.format(w.encode('utf-8'), score)
			#print '\t', w2q[w][:3]

		# show the criteria for each word based on w2ntrials
		with open('w2q.csv','w') as file:
			writer = csv.writer(file,delimiter=",")
			writer.writerow(['word', 'total','q'])
			#for w in w2q_reverse[:10]:
			for w in w2ntrials_sorted[:10]:
				#tid_list = []
				#for i in w2ntrials[w]:
				#	tid_list.append(all_trials[i])
				for q in w2q[w]:
					#print type(w), type(q)
					writer.writerow([w.encode('ascii','ignore'), len(w2q[w]), \
									q.encode('ascii','ignore')]) #+','+q+'\n')
					#print w, len(w2q[w]), q


		# filter out phrases that don't occur in other entries
		with open("tfidf_scikit.csv", "w") as file:
			writer = csv.writer(file, delimiter=",")
			writer.writerow(["DocID", "Phrase", "Score"])
		 
			doc_id = 0
			for doc in tfidf_matrix.todense():
				#print "Document %d" %(doc_id)
				#word_id = 0
				entry = doc.tolist()[0]
				phrase_scores = \
					[pair for pair in zip(range(0, len(entry)), entry) if pair[1] > 0]
				phrase_scores_sorted = sorted(phrase_scores, key=lambda t: t[1])
				for widx, score in phrase_scores_sorted:
					word = feature_names[widx]
					writer.writerow([doc_id+1, word.encode("utf-8"), score])
				#for score in doc.tolist()[0]:
				#	if score > 0:
				#		word = feature_names[word_id]
				#		writer.writerow([doc_id+1, word.encode("utf-8"), score])
				#	word_id +=1
				doc_id +=1

	return corpus, w2q, w2ntrials, wc_content2trials, all_trials, tid2q


#----------------------------------------------------------------------

def fuzzy_match(cond2trials_sorted, ratio_cutoff=90):
	# group by condition
	from fuzzywuzzy import fuzz
	import operator

	graph = defaultdict(dict)

	fuzzy_cond2cond = {}
	for i in range(len(cond2trials_sorted)):
		cond_orig = cond2trials_sorted[i]
		#print 'original:', cond_orig
		# make sure there's an entry in graph
		graph[cond_orig]={}
		# get fuzzy ratios
		w_ratios = {}
		for j in range(i+1,len(cond2trials_sorted)):
			r = fuzz.ratio(cond2trials_sorted[i], cond2trials_sorted[j])
			w_ratios[cond2trials_sorted[j]]=r
		# sort ratios
		sorted_x = sorted(w_ratios.items(), key=operator.itemgetter(1), reverse=True)
		for cond_new, s in sorted_x:
			if s>=ratio_cutoff:
				#print '\t',cond_new, s
				if cond_new not in fuzzy_cond2cond:
					fuzzy_cond2cond[cond_new]=cond_orig
			# add to graph
			if s>0:
				graph[cond_orig][cond_new]=s/100.0

	return fuzzy_cond2cond, graph

#----------------------------------------------------------------------

def build_graph(cond2trials_sorted):
	for i in range(len(cond2trials_sorted)):
		cond_orig = cond2trials_sorted[i]

#----------------------------------------------------------------------

def get_intervention(G_int, trials_criteria):
	type2trials = defaultdict(list)

	for tid in trials_criteria:
		intervention = trials_criteria[tid]['intervention']
		if not intervention: continue
		items = intervention.split('\n\n')
		for item in items:
			t, name, desc, arm_group_label, other_name = item.split('|')
			if tid not in type2trials[t]:
				type2trials[t].append(tid)
	print 'get_inervention type2trials:', type2trials

	type2ntrials_sorted = sorted(type2trials, key=lambda e: len(type2trials[e]), reverse=True)
	for t in type2ntrials_sorted: #[:20]:
		score = float(len(type2trials[t]))/len(trials_criteria)
		print '\t{0: <20} {1}'.format(t.encode('utf-8'), score*100)
		G_int[t]=score

	return type2trials, type2ntrials_sorted


#----------------------------------------------------------------------

def main():
	#get_from_web()
	#get_drug_words()
	#exit()

	# build a graph
	G_drug = defaultdict(dict)
	G_tfidf = defaultdict(dict)

	# think about number of trials contain this word
	corpus, w2q, w2ntrials_orig, wc_content2trials_orig, all_trials, tid2q = \
		get_top_words(G_drug['system'], G_tfidf['system'], trials_criteria, trials_criteria2, \
						incl_w2trials, excl_w2trials)

	# look at disease first

	# Let's look at the dependencies of this example:
	"""
	example = "acute or chronic liver, renal, or pancreas disease."
	example = "- previously untreated brain metastases. patients who have received radiation or surgery for brain metastases are eligible if therapy was completed at least 2 weeks prior to study entry and there is no evidence of central nervous system disease progression, mild neurologic symptoms, and no requirement for chronic corticosteroid therapy. enzyme-inducing anti-convulsants are contraindicated."
	parsedEx = en_parser(example.decode('utf-8'))
	# shown as: original token, dependency tag, head word, left dependents, right dependents
	for token in parsedEx:
		print(token.orth_, token.dep_, token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])
	# npadvmod = noun phrase as adv/mod

	print 'Sents:', parsedEx.sents

	disease = en_parser.vocab[u'disease']

	# cosine similarity
	from numpy import dot
	cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))


	# gather all known words, take only the lowercased versions
	allWords = list({w for w in parsedEx if w.has_vector and w.orth_.islower() and w.lower_ != "disease"})
	allWords = list(set(allWords))
	#allWords.sort(key=lambda w: cosine(w.repvec, disease.repvec))

	w2score = {}
	for w in allWords:
		s = cosine(w.vector, disease.vector)
		w2score[w]=s

	import operator
	sorted_x = sorted(w2score.items(), key=operator.itemgetter(1), reverse=True)
	print sorted_x
	"""

	# trials
	cond2trials = defaultdict(list)
	for tid in trials_criteria:
		c = trials_criteria[tid]['condition'].strip().lower()
		arr = c.split('|')
		for item in arr:
			cond2trials[item].append(tid)
	# sort
	cond2trials_sorted = sorted(cond2trials, key=lambda e: len(cond2trials[e]), reverse=True)

	#for c in cond2trials_sorted:
	#	print c, len(cond2trials[c])

	# matching conditions with 90% string comparison	
	fuzzy_cond2cond, graph = fuzzy_match(cond2trials_sorted, 90)
	# manually add mappings
	if 'colitis, ulcerative' not in fuzzy_cond2cond:
		fuzzy_cond2cond['colitis, ulcerative']='ulcerative colitis'
	if 'ulcerative colitis (uc)' not in fuzzy_cond2cond:
		fuzzy_cond2cond['ulcerative colitis (uc)']='ulcerative colitis'
	if 'ulcerative colitis, active severe' not in fuzzy_cond2cond:
		fuzzy_cond2cond['ulcerative colitis, active severe']='ulcerative colitis'
	if 'ulcerative colitis, remission (3a: with ibs symptoms, 3b: without ibs symptoms)' not in fuzzy_cond2cond:
		fuzzy_cond2cond['ulcerative colitis, remission (3a: with ibs symptoms, 3b: without ibs symptoms)']='ulcerative colitis'
	if 'ulcerative rectocolitis' not in fuzzy_cond2cond:
		fuzzy_cond2cond['ulcerative rectocolitis']='ulcerative colitis'
	if 'ibd' not in fuzzy_cond2cond:
		fuzzy_cond2cond['ibd']='inflammatory bowel disease'
	if "crohn's colitis" not in fuzzy_cond2cond:
		fuzzy_cond2cond["crohn's colitis"]="crohn's disease"
	if 'crohn' not in fuzzy_cond2cond:
		fuzzy_cond2cond['crohn']="crohn's disease"
	if "crohn's" not in fuzzy_cond2cond:
		fuzzy_cond2cond["crohn's"]="crohn's disease"
	if "complicated crohn's disease" not in fuzzy_cond2cond:
		fuzzy_cond2cond["complicated crohn's disease"]="crohn's disease"
	if 'perianal fistula' not in fuzzy_cond2cond:
		fuzzy_cond2cond['perianal fistula']="fistula"
	if 'anal fistula' not in fuzzy_cond2cond:
		fuzzy_cond2cond['anal fistula']="fistula"
	if "inflammatory bowel disease (crohn's disease and ulcerative colitis)" not in fuzzy_cond2cond:
		fuzzy_cond2cond["inflammatory bowel disease (crohn's disease and ulcerative colitis)"]='inflammatory bowel disease'
	if 'colonic neoplasms' not in fuzzy_cond2cond:
		fuzzy_cond2cond['colonic neoplasms']='colonic diseases'
	if 'diverticula' not in fuzzy_cond2cond:
		fuzzy_cond2cond['diverticula']='diverticulitis'
	if 'diarrhea' not in fuzzy_cond2cond:
		fuzzy_cond2cond['diarrhea']='chronic diarrhea'
	if 'rectal neoplasms' not in fuzzy_cond2cond:
		fuzzy_cond2cond['rectal neoplasms']='rectal cancer'
	print 'fuzzy cond match len:', len(fuzzy_cond2cond)

	# combine
	for c in fuzzy_cond2cond:
		trials = cond2trials[c]
		c_new = fuzzy_cond2cond[c]
		cond2trials[c_new]+=trials
		del cond2trials[c]

	# sort again
	cond2trials_sorted = sorted(cond2trials, key=lambda e: len(cond2trials[e]), reverse=True)

	# matching conditions with other cutoff
	#fuzzy_cond2cond, graph = fuzzy_match(cond2trials_sorted, 80)
	#print 'fuzzy cond match len:', len(fuzzy_cond2cond)

	# show

	alltype2trials = {} #defaultdict(dict)
	alltype2w2q = {}
	alltype2tid2q = {}
	realtype2tid = defaultdict(list)

	G_cond_inter_drug = defaultdict(dict)
	G_cond_inter_tfidf = defaultdict(dict)

	for c in cond2trials_sorted[:]:
		c = c.lower()
		score = len(cond2trials[c])/float(len(trials_criteria))
		print 'Condition:', c, len(cond2trials[c]), score
		# graph on condition
		G_cond_inter_drug['system'][c]=score
		#G_cond_inter_tfidf['system'][c]=score
		#cond2ntrials[c]=len(cond2trials[c])

		# graph on intervention type
		trials = cond2trials[c]
		new_trials_criteria = \
			{ k:v for k,v in trials_criteria.items() if k in trials }
		type2trials, type2ntrials_sorted = \
			get_intervention(G_cond_inter_drug[c], new_trials_criteria)
		# make a copy
		G_cond_inter_tfidf = copy.copy(G_cond_inter_drug)
		
		if not type2ntrials_sorted: continue

		#for t in type2trials:
		#	alltype2trials[t]+=type2trials[t]
		#	alltype2trials[t] = list(set(alltype2trials[t]))

		for t0 in type2ntrials_sorted:
			# look at the first type only
			#t0=type2ntrials_sorted[0]
			t0_trials = type2trials[t0]
			print 'Type:',t0, 'Num trials:', len(t0_trials)

			# get criteria ready for drug/tfidf words
			t0_new_trials_criteria = \
				{ k:v for k,v in new_trials_criteria.items() if k in t0_trials }
			t0_new_trials_criteria2 = \
				{ k:v for k,v in trials_criteria2.items() if k in t0_trials }

			t0_new_incl_w2trials = copy.copy(incl_w2trials)
			t0_new_excl_w2trials = copy.copy(excl_w2trials)
			# more on getting these ready
			for w in t0_new_incl_w2trials:
				incl_trials = t0_new_incl_w2trials[w]
				t0_new_incl_w2trials[w]=[t for t in incl_trials if t in t0_trials]
			for w in t0_new_excl_w2trials:
				excl_trials = t0_new_excl_w2trials[w]
				t0_new_excl_w2trials[w]=[t for t in excl_trials if t in t0_trials]

			# then finally get drug/tfidf words
			corpus, w2q, w2ntrials, wc_content2trials, all_trials, tid2q = \
				get_top_words(G_cond_inter_drug[c+'-'+t0], G_cond_inter_tfidf[c+'-'+t0], \
					t0_new_trials_criteria, t0_new_trials_criteria2,
					t0_new_incl_w2trials, t0_new_excl_w2trials)
			# save w2q
			alltype2w2q[c+'-'+t0]=w2q
			alltype2trials[c+'-'+t0]=t0_new_excl_w2trials
			alltype2tid2q[c+'-'+t0]=tid2q
			# store type to tids
			realtype2tid[t0]+=tid2q.keys()
			realtype2tid[t0]=list(set(realtype2tid[t0]))



	G_cond_inter_drug_system_sorted = sorted(G_cond_inter_drug['system'].items(), key=itemgetter(1), reverse=True)
	for c, v in G_cond_inter_drug_system_sorted[:10]:
		print c, ',',v

	#ptotal, words = get_words_cutoff(cond2ntrials, 0.6)


	#from dijkstra import Dijkstra, shortestPath
	#print Dijkstra(graph,"crohn's disease")
	#print shortestPath(graph,"ibd",'anemia')

	return G_cond_inter_drug, G_cond_inter_tfidf, cond2trials, alltype2w2q, \
			alltype2trials, alltype2tid2q, realtype2tid, fuzzy_cond2cond, \
			w2ntrials_orig, wc_content2trials_orig

#----------------------------------------------------------------------

def get_greedy(G, node):
	x = G[node]
	sorted_x = sorted(x.items(), key=itemgetter(1), reverse=True)
	return sorted_x

#----------------------------------------------------------------------

#COND_QUESTION = "<b>HDP:</b> What's the <b>condition</b> you are searching for? (crohn's disease, ulcerative colitis)"
COND_QUESTION = "<b>HDP:</b> Are you searching for condition <b>XX</b>?"
#INTER_QUESTION = "<b>HDP:</b> What's the preferred intervention type? (Drug, Biological, Device)"
INTER_QUESTION = "<b>HDP:</b> Is your preferred intervention type <b>XX</b>?"
KEYWORD_QUESTION = "<b>HDP:</b> To help us narrow down the trials, are you currently using <b>XX</b>?"


#----------------------------------------------------------------------

def get_trials_cond(G_cond_inter_tfidf, cond=None):
	found_in_db = False
	ret_str = ''
	num_cond_q_asked = 0
	remainder = []
	#
	all_cond = G_cond_inter_tfidf['system'].keys()
	if not cond:
		# ask user questions, starting with the most probable one
		for c, v in get_greedy(G_cond_inter_tfidf, 'system'):
			next_q_cond = re.sub('XX', c, COND_QUESTION)
			rsp = raw_input(next_q_cond+'\n').strip().lower()
			num_cond_q_asked+=1 # keep track of number of cond q asked
			if rsp[0]=='y': 
				found_in_db = True
				break
		if found_in_db:
			remainder = cond2trials[c]
			ret_str = "You've selected "+c+', which narrows down to '+str(len(remainder))+\
				' trials ('+str(v)+'% of all trials)'
		else:
			c = cond
			remainder = []
			v = 0.0
			ret_str = 'We apologize. We did not find the condition you are searching for.'
	#
	elif cond not in all_cond:
		# cond not in database
		#print 'NOTE. Condition', cond, 'not in database'
		found_in_db = False
		c = cond
		remainder = []
		v = 0.0
		ret_str="We apologize. You've selected "+c+', which is not in our database.'
	#
	else:
		# cond is in database
		found_in_db = True
		c = cond
		remainder = cond2trials[c]
		v = len(remainder)/float(len(trials_criteria))
		ret_str = "You've selected "+c+', which narrows down to '+str(len(remainder))+\
			' trials ('+str(v)+'% of all trials)'
		# see how far into the ordered list it is
		# i.e., number of questions it would've taken to get there
		greedy_list = get_greedy(G_cond_inter_tfidf, 'system')
		greedy_list = [cc for cc, vv in greedy_list]
		num_cond_q_asked = greedy_list.index(c)
	#
	return ret_str, c, v, found_in_db, num_cond_q_asked, remainder


#----------------------------------------------------------------------

def get_trials_type(G_cond_inter_tfidf, c, intertype=None):
	found_in_db = False # reset here
	ret_str = ''
	num_type_q_asked = 0
	remainder = []

	all_type = G_cond_inter_tfidf[c].keys()
	if not intertype:
		# ask user questions, starting with the most probable one
		for t, v2 in get_greedy(G_cond_inter_tfidf, c):
			next_q_t = re.sub('XX', t, INTER_QUESTION)
			rsp = raw_input(next_q_t+'\n').strip().lower()
			num_type_q_asked+=1
			if rsp[0]=='y': 
				found_in_db = True
				break
		if found_in_db:
			c_t = c+'-'+t
			remainder = alltype2tid2q[c_t].keys()
			ret_str = "You've selected "+c+'-'+t+', which narrows down to'+ \
				str(len(remainder))+' trials ('+str(v2)+'% of all trials)'
		else:
			t = intertype
			v2 = 0.0
			print 'We apologize. We did not find intervention type,', \
					intertype

	elif intertype not in all_type:
		# intervention type not available
		found_in_db = False
		t = intertype
		v2 = 0.0
		ret_str = "We apologize. You've selected intervention type, "+t+ \
				", which is not in our database."
		#print "We apologize. You've selected", c, '-', t, ', \
		#		which is not in our database.'

	else:
		# specific condition is given and in database
		found_in_db = True
		t = intertype
		c_t = c+'-'+t
		remainder = alltype2tid2q[c_t].keys()
		v2 = len(remainder)/float(len(trials_criteria))
		ret_str = "You've selected "+c+'-'+t+', which narrows down to '+ \
				str(len(remainder))+' trials ('+str(v2)+'% of all trials)'
		# see how many questions we would've asked about intervention type
		greedy_list = get_greedy(G_cond_inter_tfidf, c)
		greedy_list = [tt for tt, vv in greedy_list]
		num_type_q_asked = greedy_list.index(t)

	return ret_str, t, v2, found_in_db, num_type_q_asked, remainder


#----------------------------------------------------------------------

def get_trials_word(G_cond_inter_tfidf, c_t, excl_word=None):
	found_in_db = False # reset here
	ret_str = ''
	num_word_q_asked = 0
	remainder = []

	all_keyword = G_cond_inter_tfidf[c_t].keys()
	if not excl_word:
		# ask user questions, starting with the most probable one
		all_avoid_trials = []
		for w, vv in get_greedy(G_cond_inter_tfidf, c_t):
			# see how many trials are being excluded
			avoid_trials = set(alltype2tid2q[c_t].keys()) & set(excl_w2trials[w])
			if len(avoid_trials)==0: continue

			num_word_q_asked+=1
			next_q_w = re.sub('XX', w, KEYWORD_QUESTION)
			rsp = raw_input(next_q_w+'\n').strip().lower()

			if rsp[0]=='y':
				# store trials excluded
				#avoid_trials = set(alltype2tid2q[c_t].keys()) & set(excl_w2trials[w])
				found_in_db = True
				all_avoid_trials += list(avoid_trials)
				remainder = set(alltype2tid2q[c_t].keys())-set(all_avoid_trials)
				v3 = len(remainder)/float(len(trials_criteria))
				break
			# see how many trials are left
			remainder = set(alltype2tid2q[c_t].keys())-set(all_avoid_trials)
			v3 = len(remainder)/float(len(trials_criteria))
			print 'remaining trials:', len(remainder), '(',v3,'of all trials)'

		if found_in_db:
			print "You've selected",c_t, '-', w, ', which narrows down to ', \
				len(remainder), ' trials (', v3,'% of all trials)'
		else:
			w = excl_word
			v3 = 0.0
			print 'We apologize. We did not find exclusion criteria,', excl_word
	
	elif excl_word not in all_keyword:
		# intervention type not available
		found_in_db = False
		w = excl_word
		v3 = 0.0
		print "We apologize. Your exclusion criteria,", w, ", is not in our database."		
		#print "We apologize. You've selected", c, '-', t, '-', w, ', which is not in our database.'

	else:
		# found in database
		found_in_db = True
		w = excl_word
		all_avoid_trials = set(alltype2tid2q[c_t].keys()) & set(excl_w2trials[w])
		remainder = set(alltype2tid2q[c_t].keys())-set(all_avoid_trials)
		v3 = len(remainder)/float(len(trials_criteria))
		print "You've selected",c_t, '-', w, ', which narrows down to ', \
				len(remainder), ' trials (', v3,'% of all trials)'
		# find out how many questions about words we would've asked
		greedy_list = get_greedy(G_cond_inter_tfidf, c_t)
		greedy_list = [ww for ww, vv in greedy_list]
		num_word_q_asked = greedy_list.index(w)

	return ret_str, w, v3, found_in_db, num_word_q_asked, remainder


#----------------------------------------------------------------------

def get_trials(G_cond_inter_tfidf, cond=None, intertype=None, excl_word=None):

	"""
	print "----> get_trials; you've entered:"
	print '\tcond:', cond
	print '\tintertype:', intertype
	print '\texcl_word:', excl_word
	"""

	num_cond_q_asked = 0
	num_type_q_asked = 0
	num_word_q_asked = 0

	trials_cond_remainder = []
	trials_type_remainder = []
	
	ntrials = 1
	remainder = []

	#--------------------------------------------------
	# condition type
	ret_str, c, v, found_in_db, num_cond_q_asked, remainder, \
		= get_trials_cond(G_cond_inter_tfidf, cond=cond)

	trials_cond_remainder.append(copy.copy(remainder))

	if not found_in_db or len(remainder)<=ntrials:
		t = intertype
		v2 = 0.0
		w = excl_word
		v3 = 0.0
		return (c,v), (t,v2), (w,v3), remainder, num_cond_q_asked, \
			num_type_q_asked, num_word_q_asked, trials_cond_remainder, \
			trials_type_remainder


	#--------------------------------------------------
	# intervention type
	ret_str, t, v2, found_in_db, num_type_q_asked, remainder, \
		= get_trials_type(G_cond_inter_tfidf, c, intertype=intertype)

	c_t = c+'-'+t
	
	trials_type_remainder.append(copy.copy(remainder))

	if not found_in_db or len(remainder)<=ntrials:
		w = excl_word
		v3 = 0.0
		return (c,v), (t,v2), (w,v3), remainder, num_cond_q_asked, \
				num_type_q_asked, num_word_q_asked, trials_cond_remainder, \
				trials_type_remainder


	#--------------------------------------------------
	# exclusion keyword
	ret_str, w, v3, found_in_db, num_word_q_asked, remainder = \
		get_trials_word(G_cond_inter_tfidf, c_t, excl_word=excl_word)

	return (c,v), (t,v2), (w,v3), remainder, num_cond_q_asked, \
			num_type_q_asked, num_word_q_asked, trials_cond_remainder, \
			trials_type_remainder

#----------------------------------------------------------------------

def test_avg(n, wc_content2trials, wc_content2trials_sorted2):
	# now try 3 words
	w_asked2 = []
	w_asked2_trials_left = []
	for i in range(1000):
		wlist = random.sample(wc_content2trials_sorted2,n)
		idxes = []
		trials2 = []
		for w in wlist:
			idxes.append(wc_content2trials_sorted2.index(w))
			# store actual trials
			trials2+=wc_content2trials[w]
		# question is furthest into list
		q = max(idxes)
		w_asked2.append(q)
		# remove duplicates
		trials2 = list(set(trials2))
		trials_left = len(trials_criteria)-len(trials2)
		w_asked2_trials_left.append(trials_left)
	#
	avg_w_asked2 = sum(w_asked2)/float(len(w_asked2))
	#print avg_w_asked2
	avg_w_asked2_trials_left = sum(w_asked2_trials_left)/float(len(w_asked2_trials_left))
	#print avg_w_asked2_trials_left
	return w_asked2, w_asked2_trials_left, avg_w_asked2, avg_w_asked2_trials_left

#----------------------------------------------------------------------

if __name__ == '__main__':
	G_cond_inter_drug, G_cond_inter_tfidf, cond2trials, alltype2w2q, \
		alltype2trials, alltype2tid2q, realtype2tid, fuzzy_cond2cond, \
		w2ntrials, wc_content2trials = main()

	# store as pickle files
	pickle.dump(G_cond_inter_tfidf, open('G_cond_inter_keyword.pickle','wb'))
	#pickle.dump(G_cond_inter_drug, open('G_cond_inter_drug.pickle','wb'))
	pickle.dump(alltype2tid2q, open('alltype2tid2q.pickle','wb'))
	pickle.dump(cond2trials, open('cond2trials.pickle','wb'))

	# get one path: ask user
	(c,pc), (t,pt), (w,p2), tids, \
	num_cond_q_asked, num_type_q_asked, num_word_q_asked, \
	trials_cond_remainder, trials_type_remainder = \
		get_trials(G_cond_inter_tfidf)

	total = 0
	for t in realtype2tid:
		#print t, len(realtype2tid[t])
		total+=len(realtype2tid[t])
	print 'total:', total
	for t in realtype2tid:
		print t, len(realtype2tid[t]), len(realtype2tid[t])/float(total)

	wc2 = Counter()
	wc2_tids = defaultdict(list)
	more_tid2q = alltype2tid2q["crohn's disease-Drug"]
	for tid in list(tids):
		#print tid, len(more_tid2q[tid])
		content = ' '.join(more_tid2q[tid])
		tokens = tokenize_text(content)
		for t in tokens:
			wc2[t]+=1
			wc2_tids[t].append(tid)
			wc2_tids[t] = list(set(wc2_tids[t]))


	wc2_tids_sorted = sorted(wc2_tids, \
		key=lambda e: len(wc2_tids[e]), reverse=True)

	for w in wc2_tids_sorted[:10]:
		if w in ['study', 'disease', 'patient', 'history', 'prior', \
				'drug', 'use']: continue
		print w, len(wc2_tids[w]), len(wc2_tids[w])/float(len(tids))
		print '\t', wc2_tids[w]


	#ordered_conditions = sorted(G_cond_inter_drug['system'].keys())
	#csvwriter = csv.writer(open('conditions.csv','wb'))
	#csvwriter.writerow(['condition'])
	#for cond in ordered_conditions:
	#	csvwriter.writerow([cond])


	cond2trials_sorted = sorted(cond2trials, key=lambda e: len(cond2trials[e]), reverse=True)
	for cond in cond2trials_sorted[:30]:
		print cond, len(cond2trials[cond])

	# try introducing probabilities based on disease prevalence

	prevalence = {"crohn's disease":0.002,
		'ulcerative colitis':0.002,
		'inflammatory bowel disease':0.002,
		'rheumatoid arthritis':0.006,
		'ankylosing spondylitis':0.01,
		'colorectal cancer':0.0004,
		'irritable bowel syndrome':0.15,
		'psoriasis':0.03,
		'psoriatic arthritis':0.0015,
		'colorectal neoplasms': 0.0037,
		'intestinal diseases': 0.19,
		'primary sclerosing cholangitis': 0.00001,
		'gastrointestinal diseases': 0.19,
		'depression': 0.15,
		'colonic neoplasms': 0.06,
		'healthy': 0.1,
		'indeterminate colitis': 0.015,
		'graft versus host disease': 0.00005,
		'iron deficiency anemia':0.016,
		'rectal cancer':0.004,
		'chronic diarrhea':0.05,
		'abdominal pain':0.325,
		"behcet's disease":0.000066,
		'intestinal tuberculosis':0.001,
		'diverticulitis':0.2,
		'rectal neoplasms':0.06,
		'cancer':0.045,
		'fistula':0.00029,
		'clostridium difficile':0.2*0.05,
	}

	# random sampling for testing performance

	condition_pop = []
	prevalence_pop = {}
	for c in prevalence:
		count = int(math.ceil(prevalence[c]*10000))
		prevalence_pop[c]=count
		condition_pop+=[c]*count

	print prevalence_pop.values()

	total_people = sum(prevalence_pop.values())
	print 'total people:', total_people

	ntypes = len(realtype2tid)
	section = total_people/ntypes

	types_pop = realtype2tid.keys() * section
	# if there's extra, duplicate the last one
	if len(types_pop)<total_people:
		#types_pop+=[types_pop[-1]]*(total_people-len(types_pop))
		n = total_people-len(types_pop)
		idx = [random.randint(0,n) for i in xrange(n)]
		for i in idx:
			types_pop.append(types_pop[i])

	# sanity check
	if len(types_pop)!=len(condition_pop):
		print 'ERROR. Unequal number of condition and type'


	# exclusion keywords
	wc_content2trials_sorted = sorted(wc_content2trials, key=lambda e: len(wc_content2trials[e]), reverse=True)

	drugbank = codecs.open('drugbank-full-database.xml',encoding='utf-8').read().lower()

	words_new = list(wc_content2trials_sorted)
	words_drug = []

	if not os.path.isfile('words_drug.pickle'):
		for w in wc_content2trials_sorted:
			# skip words that are only numbers
			if re.search('\d+',w) and w in words_new: words_new.remove(w)
			if u"<name>"+w+u"</name>" in drugbank:
				words_drug.append(w)
		pickle.dump(words_drug, open('words_drug.pickle','wb'))

	words_drug = pickle.load(open('words_drug.pickle'))

	part = len(condition_pop)/len(words_drug)
	words_pop = words_drug * part
	# if there's extra, duplicate the last one
	if len(words_pop)<total_people:
		n = total_people-len(words_pop)
		idx = [random.randint(0,n) for i in xrange(n)]
		for i in idx:
			words_pop.append(words_drug[i])

	# sanity check
	if len(types_pop)!=len(condition_pop)!=len(words_pop):
		print 'ERROR. Unequal number of condition/type/word'

	# randomize the condition and intervention types
	random.shuffle(condition_pop)
	random.shuffle(types_pop)
	random.shuffle(words_pop)



## continuing from __main__; copy and paste to command line for testing

rec_trials = []
combo = []
num_c_q_asked = []
num_t_q_asked = []
num_w_q_asked = []
num_q_asked = []

num_c_trials_remain = []
num_t_trials_remain = []

for i in range(len(condition_pop)):
	cnew = condition_pop[i]
	tnew = types_pop[i]
	wnew = words_pop[i]
	(c,v), (t,v2), (w,v3), ts, \
	num_cond_q_asked, num_type_q_asked, num_word_q_asked, \
		trials_cond_remainder, trials_type_remainder = \
			get_trials(G_cond_inter_tfidf, cond=cnew, intertype=tnew, excl_word=wnew)
	# store results
	combo.append(cnew+'-'+tnew+'-'+wnew)
	rec_trials.append(ts)
	num_c_q_asked.append(num_cond_q_asked)
	num_t_q_asked.append(num_type_q_asked)
	num_w_q_asked.append(num_word_q_asked)
	num_q_asked.append(num_cond_q_asked+num_type_q_asked+num_word_q_asked)
	#
	num_c_trials_remain.append(len(trials_cond_remainder))
	num_t_trials_remain.append(len(trials_type_remainder))
	# test
	#if 'crohn' in cnew and 'Drug' in tnew and 'infliximab' in wnew:
	#	print cnew, tnew, wnew, ':', len(ts)

n_rec_trials = [len(tlist) for tlist in rec_trials]

avg_trials = sum(n_rec_trials)/float(len(n_rec_trials))
print 'avg trials left:', avg_trials
avg_q = sum(num_q_asked)/float(len(num_q_asked))
print 'avg questions asked:', avg_q
avg_cq = sum(num_c_q_asked)/float(len(num_c_q_asked))
avg_tq = sum(num_t_q_asked)/float(len(num_t_q_asked))
avg_wq = sum(num_w_q_asked)/float(len(num_w_q_asked))
print 'avg c, t, w questions asked:', avg_cq, avg_tq, avg_wq

avg_cr = sum(num_c_trials_remain)/float(len(num_c_trials_remain))
print 'avg trials left after asking cond:', avg_cr
avg_tr = sum(num_t_trials_remain)/float(len(num_t_trials_remain))
print 'avg trials left after asking type:', avg_tr


#------ naive approach: drug-words only
wc_content2trials_sorted2 = copy.copy(wc_content2trials_sorted)
for www in wc_content2trials_sorted:
	if www not in words_drug:
		wc_content2trials_sorted2.remove(www)


# original dict mapping words to list of trials: wc_content2trials[w]
print 'num of all words:', len(wc_content2trials_sorted)
print 'num of drug-related words:', len(wc_content2trials_sorted2)

xx_all_trials = list(set(trials_criteria.keys()))
half_trials = len(xx_all_trials)/2
print 'half trials:', half_trials

#### sample one word
w_asked = []
w_asked_trials_left = []
for i in range(1000):
	[wnew] = random.sample(wc_content2trials_sorted2,1)
	q = wc_content2trials_sorted2.index(wnew)
	w_asked.append(q)
	# trials left after this exclusion word
	trials1 = wc_content2trials[wnew]
	trials1 = list(set(trials1)) # remove duplicates
	trials_left = len(trials_criteria)-len(trials1)
	w_asked_trials_left.append(trials_left)

avg_w_asked = sum(w_asked)/float(len(w_asked))
print avg_w_asked
avg_w_asked_trials_left = sum(w_asked_trials_left)/float(len(w_asked_trials_left))
print avg_w_asked_trials_left


#### sample different number of words
for n in range(2,151,10):
	w_askedX, w_askedX_trials_left, avg_w_askedX, avg_w_askedX_trials_left = \
		test_avg(n, wc_content2trials, wc_content2trials_sorted2)
	print n, 'words:', avg_w_askedX, 'avg q asked;', avg_w_askedX_trials_left, 'avg trials left'


#### sample from most frequent to least frequent
collect_trials = []
num_words = 0
for i, w in enumerate(wc_content2trials_sorted2):
	collect_trials += wc_content2trials[w]
	collect_trials = list(set(collect_trials)) # remove duplicates
	if len(collect_trials)>=half_trials: break

print 'num of words needed to reach at least half trials:', i
print 'num of trials reached:', len(collect_trials)


########

# ask individual condition, intervention type, etc.
ret_str, c, v, found_in_db, num_cond_q_asked, remainder = \
	get_trials_cond(G_cond_inter_tfidf, cond='diverticulosis, colonic') #"inflammatory bowel disease") #ulcerative colitis") #"crohn's disease")

print ret_str
print 'num q asked:', num_cond_q_asked
print 'remaining trials:', len(remainder), len(re)

ret_str, t, v2, found_in_db, num_type_q_asked, remainder, \
		= get_trials_type(G_cond_inter_tfidf, c, intertype="Drug")

print ret_str
print 'num q asked:', num_type_q_asked
print 'remaining trials:', len(remainder)

# conditions and associated trials: cond2trials
# types and associated tids: realtype2tid

tuples = get_greedy(G_cond_inter_tfidf, 'system')

for w, vv in tuples[:10]:
	ret_str, c, v, found_in_db, num_cond_q_asked, remainder = \
		get_trials_cond(G_cond_inter_tfidf, cond=w)
	print w
	#print ret_str
	print 'num q asked:', num_cond_q_asked, \
		'remaining trials:', len(remainder), \
		float(len(remainder))/len(trials_criteria), \
		1-float(len(remainder))/len(trials_criteria)


ret_str, t, v2, found_in_db, num_type_q_asked, remainder, \
		= get_trials_type(G_cond_inter_tfidf, c, intertype="Drug")

print ret_str
print 'num q asked:', num_type_q_asked
print 'remaining trials:', len(remainder)


