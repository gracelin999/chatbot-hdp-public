
import sys, os
import re, csv, codecs
from collections import defaultdict

from datetime import date
from quantulum import parser as quan_parser
#from spacy.en import English

import psycopg2
import pandas as pd
from fuzzywuzzy import fuzz
# import sputnik
# import spacy.about
import urllib2

THISPATH = os.path.dirname(os.path.realpath(__file__))

#---------------------------------------------------------------------------

#### if you want to use spacy and/or drugbank, uncomment the following

#print 'Loading parser and drugbank...',
#en_parser = English()
drugbank = []
#drugbank = codecs.open('drugbank-full-database.xml',encoding='utf-8').read().lower()
# urllib2.urlopen('https://s3-us-west-2.amazonaws.com/drug-bank/drugbank-full-database.xml').read().lower()
#print 'Done.\n'


MED_LEXICON = codecs.open(os.path.join(THISPATH,'med_lexicon.txt'),encoding='utf-8').read().split('\n')
lexicon_append = open(os.path.join(THISPATH,'med_lexicon.txt'), 'ab')

month_map = {
	'january':1, 'jan':1,
	'february':2, 'feb':2,
	'march':3,
	'april':4,
	'may':5,
	'june':6,
	'july':7,
	'august':8, 'aug':8,
	'september':9, 'sept':9,
	'october':10, 'oct':10,
	'november':11, 'nov':11,
	'december':12, 'dec':12}

MILES_PER_HOUR = 35 
ANYWHERE_MILES = 24901

PEOPLE_DB = 'chatbot_people'

#---------------------------------------------------------------------------

def get_spacy_entities_only(parsedData):
	entities = {}
	ents = list(parsedData.ents)
	for entity in ents:
		entities[entity.label_]=' '.join(t.orth_ for t in entity)
	return entities


def get_spacy_info(line):
	parsedData = en_parser(line)
	# entities only
	entities = get_spacy_entities_only(parsedData)
	return entities

#---------------------------------------------------------------------------

def calculate_age(year, month, day):
    today = date.today()
    return today.year - year - ((today.month, today.day) < (month, day))

#---------------------------------------------------------------------------

def get_name(line, entities=None):
	#print entities
	if entities and 'PERSON' in entities:
		name = entities['PERSON']
	else:
		name = line
		#m = re.search('i[t\']*s (.+)',line)
		m = re.search('(names|name\'s|name is)\s*(.+)',name)
		m2 = re.search('(it is|its|it\'s)\s*(.+)',name)
		m3 = re.search('(im|am|i\'m)\s*(.+)',name)
		if m:
			name = m.group(2)
		elif m2:
			name = m2.group(2)
		elif m3:
			name = m3.group(2)

	return name

#---------------------------------------------------------------------------

def get_gender(line):
	gender = 'neither' # default

	found = False
	if line:
		arr = line.split(' ')
		for w in arr:
			w = w.strip().lower()
			if re.match('^m', w):
				gender = 'male'; found = True
			elif re.match('^f',w):
				gender = 'female'; found = True
			elif re.match('^n',w):
				gender = 'neither'; found = True
			elif re.match('^b',w):
				gender = 'both'; found = True
			
			if found:
				break

	return gender

#---------------------------------------------------------------------------

def convert_year(year_str):
	year = None
	if len(year_str)==2:
		year=int('20'+year_str)
	else:
		year=int(year_str)
	# convert
	if year>date.today().year:
		year-=100
	return year

def get_birthdate(line_orig, entities=None):
	year = None
	month = None
	day = None

	line = line_orig
	line = re.sub(',',' ',line) # replace comma
	#print line

	if '/' in line:
		m = re.search('(\d+)/(\d+)/(\d+)',line)
		month = int(m.group(1))
		day = int(m.group(2))
		year_str = m.group(3)
		# convert to 4 digits if it's 2 digits
		year = convert_year(year_str)

	if not year:
		m = re.search('(\d\d\d\d)',line)
		if m: 
			year = m.group(1)
			# remove year from line
			line = re.sub(year,'',line).strip()

	# check month and day
	arr = line.split(' ')

	if not month:
		for a in arr:
			if a.lower() in month_map.keys():
				month = month_map[a.lower()]
				# remove month from array
				arr.remove(a)
				break

	if not day:
		for a in arr:
			m = re.search('(\d+)',a)
			if m: 
				day = m.group(1)
				break

	if not year:
		m = re.search('(\d+)',arr[-1]) # check last entry
		if m:
			year_str = m.group(1)
			# convert to 4 digits if it's 2 digits
			year = convert_year(year_str)

	# convert to int
	if year: year = int(year)
	if month: month = int(month)
	if day: day = int(day)

	return [year, month, day]

#---------------------------------------------------------------------------

def get_distance(line_orig, entities=None):
	miles_per_hour = MILES_PER_HOUR
	anywhere_miles = ANYWHERE_MILES

	miles = miles_per_hour # default

	line = line_orig.lower()
	quants = quan_parser.parse(line)
	if quants:
		# look at the first quantity only 
		# TODO: other quantities?
		if quants[-1].unit.name in ['dimensionless', 'mile']:
			miles = quants[-1].value
		elif quants[-1].unit.name == 'hour':
			# convert with miles per hour
			miles = quants[-1].value * miles_per_hour
	elif ("can't" in line) or ("cannot" in line) or ("can not" in line):
		miles = miles_per_hour
	elif 'not sure' in line:
		miles = miles_per_hour
	elif 'depend' in line:
		miles = miles_per_hour
	elif 'anywhere' in line:
		miles = anywhere_miles

	return miles

#---------------------------------------------------------------------------

def get_treatments(line):
	nouns = []
	treatments = []

	# get nouns
	parsed_data = en_parser(line.decode('utf-8'))
	for i, token in enumerate(parsed_data):
		#print token.tag_
		if 'NN' in token.tag_ and token.lower_ not in nouns: #token.pos_ in ['PROPN','NOUN']:
			nouns.append(token.lower_)
	#print '\t', nouns

	# check if it's in local dic or drugbank
	for noun in nouns:
		if noun in MED_LEXICON:
			treatments.append(noun)
		elif u"<name>"+noun+u"</name>" in drugbank:
			#print 'found in drugbank'
			#a,b= re.search("<name>"+str(noun)+"</name>",drugbank).span()
			#print drugbank[a-100:b+100]
			treatments.append(noun)
			# add to lexicon and write to file
			if noun not in MED_LEXICON:
				MED_LEXICON.append(noun)
				lexicon_append.write(noun+'\n')

	# check to see if there's a typo: compare with words in current lexicon
	# TODO: change this to drugbank lexicon for more completeness
	if not treatments:
		for token in parsed_data:
			w = token.lower_
			for wmed in MED_LEXICON:
				# compare strings and return similarity ratio
				r = fuzz.ratio(w, wmed)
				#print w, wmed, r
				if r > 85: # if they are above certain level of similar, we'll take it
					if wmed not in treatments:
						treatments.append(wmed)
					break

	return [str(t) for t in treatments]

#---------------------------------------------------------------------------

def get_yesno(line):
	if line.lower()[0]=='y':
		return 'yes'
	else:
		return 'no'

#---------------------------------------------------------------------------


def insert_table_query(id, name, gender, birth_year, birth_month, birth_day,
		age, dist_miles, treatments, rsp_list):

	if not treatments:
		treatments_str = 'None'
	else:
		treatments_str = ','.join(treatments)

	rsp_list_str = '|'.join(rsp_list)

	query = "insert into "+PEOPLE_DB+" (id, name, gender, birth_year, "+\
			"birth_month, birth_day, dist_miles, treatments, raw_rsp) values "+\
			"("+str(id)+", \'"+name+"', '"+gender.lower()+"', "+\
			str(birth_year)+", "+str(birth_month)+","+str(birth_day)+", "+\
			str(dist_miles)+", '"+treatments_str+"', '"+rsp_list_str+"');"
	#print query

	return query

#---------------------------------------------------------------------------

def get_query(min_age=26, max_age=26, miles=75, gender='Male'):
	op = ''
	if gender.lower() == 'male':
		op = "trials.gender = 'Male' or "
	elif gender.lower() == 'female':
		op = "trials.gender = 'Female' or "
	op += "trials.gender = 'Both'"

	op2 = ''
	if miles!=ANYWHERE_MILES:
		meters = miles/0.621371*1000 # conversion to meters
		op2 = "and ST_DWithin(ST_FlipCoordinates(locations.point)::geography, " \
				+"ST_FlipCoordinates(ST_GeomFromText('POINT(37.7749 -122.4194)',4326))::geography, "+str(meters)+") "

	# create query with age, gender, miles, etc.
	# actual query omitted here since it's not used in the simplified version
	sql_query = ""
	return sql_query


#---------------------------------------------------------------------------

def parse_all_conv(max=None):
	# parse simulated conversations

	hdp_questions = {
		1:'name',
		2:'gender',
		3:'birthdate',
		4:'distance willing to travel (in miles)',
		5:'treatments so far'}

	words_per_turn = []
	turns_per_person = []
	turns_per_hdp = []

	total_conv = 0

	names_found = 0
	gender_found = 0
	bdate_found = 0
	dist_found = 0
	treatments_found = 0

	persons = defaultdict(dict)

	for f in os.listdir('conversations')[:max]:
		if f[0]=='.': continue # skip hidden file

		m = re.search('conversation(\d+)',f)
		id = m.group(1)

		print '\n----',f
		total_conv+=1

		hdp_count = 0
		person_count = 0

		reader = csv.reader(open(os.path.join('conversations',f),'rb'))
		next(reader) # skip header

		for row in reader: 
			if row[0]=='Person':
				person_count+=1
				if len(row)>1:
					#print len(arr[1])
					words = row[1].split(' ')
					words = [w for w in words if w]
					#print len(words)
					words_per_turn.append(len(words))
				
				print 'Q:',hdp_questions[hdp_count]
				print row[1]

				entities = get_spacy_info(row[1].decode('utf-8'))
				print '\tall entities:', entities

				# ----- get name -----
				if hdp_count==1:
					if 'name' in persons[id] and persons[id]['name']:
						continue

					name = get_name(row[1], entities)
					if name:
						print '\tFOUND:', name
						names_found+=1
					persons[id]['name']=name

				# ----- get gender -----
				elif hdp_count==2:
					gender = get_gender(row[1])
					if gender:
						print '\tFOUND:', gender
						gender_found+=1
					persons[id]['gender']=gender

				# ----- get birthdate -----
				elif hdp_count==3:
					[year, month, day] = get_birthdate(row[1], entities)
					print '\tYEAR:', year
					print '\tMONTH:', month
					print '\tDAY:', day
					if year and month and day:
						bdate_found+=1
					persons[id]['birth year']=year
					persons[id]['birth month']=month
					persons[id]['birth day']=day

				# ----- get distance (in miles) -----
				elif hdp_count==4:
					miles = get_distance(row[1],entities)
					print '\tMILES:', miles
					if miles!=None:
						# if the current one is more restrictive, 
						# then use the current version
						if 'miles' in persons[id]:
							if miles < persons[id]['miles']:
								persons[id]['miles']=miles
						else:
							dist_found+=1
					persons[id]['miles']=miles

				# ----- get treatments -----
				elif hdp_count==5:
					treatments = get_treatments(row[1])
					print '\tTREATMENTS:', treatments
					if treatments:
						treatments_found+=1
					persons[id]['treatments']=treatments

				# ----- default -----
				else:
					print '\tall entities:', entities

			elif row[0]=='HDP':
				hdp_count+=1
			else:
				print 'ERROR. Unknown row:', row

		turns_per_person.append(person_count)
		turns_per_hdp.append(hdp_count)

	return persons, words_per_turn, total_conv, names_found, gender_found, \
			bdate_found, dist_found, treatments_found


#---------------------------------------------------------------------------

# connect to db
"""
dbname = ''
username = ''
pswd = ''
host = ''
port = 

# connect
con = None
con = psycopg2.connect(host=host, port=port, dbname=dbname, \
		user=username, password=pswd)
"""


if __name__ == "__main__":

	if False:
		# run simulated documents
		# might want to redirect output to a file
		num_conv = 10
		main_sim(con, num_conv)
	else:
		# user i/f
		while True:
			print '\n'
			try:
				main(con)
			except KeyboardInterrupt:
				print "\nLater!"
				sys.exit()


	if con:
		con.close()
	



