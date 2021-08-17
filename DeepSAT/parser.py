def parse_file(path):
	f = open(path,"r")
	contents = f.readlines()

	CNF = []
	variables = []

	for line in contents:
		c = []
		for token in line.split(): 
			if token == 'c':
				break
			if token == 'p':
				variables = int(line.split()[2])
				break
			else:
				if token == '0':
					break
				else:
					c.append(int(token))
		if len(c) != 0:
			CNF.append(c)


	return variables, CNF