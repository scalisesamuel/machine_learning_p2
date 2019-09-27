# Samuel Scalise
# Undergraduate
# Python
# Project 2 Machine Learning
# Adaboost
# Need Python standard add-on csv(comes with every installation of Python >= 2.7)
#   CSV is just the standard comma separated format for text files
# Summary:
#       This program runs an adaboost algorithm on a training set to classify a decision stump

#   import random is needed to choose random numbers
#   import csv needed to read in csv, comma separated values, from a training set txt file
#   import math is needed to perform log and ln operations
import csv
import random
import math

#   title contains column names(features and classification)
#   rows contains the list of values in each row
#   training is the user inserted training set data file
title = []
rows = []
training = input("input training data file name\n")

#   Next we open up the training set data file and use csv_reader from import csv to read in the txt file values
with open(training) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            for item in row:
                title.append(item)
        else:
            rows.append(row)
            line_count += 1

#   Now we use our newly recorded input to find the total number of rows
row_count = 0
for row in rows:
    row_count += 1

#   Now we use our newly recorded input to find the total number of columns
column_count = 0
for column in title:
    column_count += 1

#   Finally we use our newly recorded input to find the position in each row that holds the classification information
#   This is assumed to be the last column based on out of program user defined format restrictions on input data
classification_pos = column_count - 1

n = 1000
i = 0
#   This whole process will go n times and repeat from the beginning if there is ever an error greater than 0.5
while i < n:
    #   This if block is the set up to start adaboost
    #   Needed in the loop so that if the loop is reset the whole process begins anew
    if i == 0:
        #   D is the list containing a Dt list for each iteration
        #   E is the list containing an epsilon(error) for each iteration
        #   A is the list containing an alpha for each iteration
        #   W is the list containing a w(weight) for each iteration
        #   n is the number of iterations. This could be many things, but for this assignment I'll go with 100
        #   h is the hypothesis (model)
        D = []
        h = []
        E = []
        A = []
        W = []

        for pos in range(0, n + 1):
            D.append([])
            E.append([])
            A.append([])
            W.append([])
            h.append([])

        #   Now it is assumed that there is only two columns from the training set because we are working with decision stumps
        #   The first column is assumed to be the feature values
        #   The second column is assumed to be the classification values
        #   Knowing these assumptions, We create the original D
        D_count = 0
        for value in rows:
            D[0].append(value)
            D_count += 1

        #   Now we calculate the weights for the first round
        #   This is 1/(The number of values in the feature)

        for value in D[0]:
            W[0].append(1/D_count)

    #   Now we have set up the loop for each iteration
    #   This loop will go for n iterations
    #   we need to loop the first part because we can get iterations of only one class
    #       This makes it imposable to make a hypothesis so we loop until that does not happen
    #   First we need to use the weights to find the values we're going to use
    #   iteration will hold info needed for this iteration of adaboost
    #       [D_position, D_value, D_class, D_predict, miss/correct predict]
    #   iteration_size is used to keep track of how many values we have selected
    error_flg = 0
    while error_flg == 0:
        error_flg = 1
        hypo_flg = 0
        while hypo_flg == 0:
            iteration = []
            iteration_size = 0

            #   This loop continues until D_count values have been selected
            #   The inner loop goes through each weight and rolls a random number 1-100
            #       If the random number is less than or equal to the weighted probability of the D element,
            #       then the D element is added to the iteration
            while iteration_size < D_count:
                pos = 0
                for value in W[i]:
                    chance = random.randint(0, 100)
                    if (chance <= (float(value) * 100)) and (iteration_size < D_count):
                        iteration.append([pos, D[0][pos][0], D[0][pos][1]])
                        iteration_size += 1
                    pos += 1

            #   Now we need to sort the iteration
            sort_flg = 0
            while sort_flg == 0:
                sort_flg = 1
                temp = []
                for count1 in range(0, D_count):
                    for count2 in range(count1, D_count):
                        if iteration[count1][1] > iteration[count2][1]:
                            for item in iteration[count1]:
                                temp.append(item)
                            for loc in range(0, 3):
                                iteration[count1][loc] = iteration[count2][loc]
                                iteration[count2][loc] = temp[loc]
                            sort_flg = 0

            #   Now we need to find the values where a sign switch happens
            #   splits are the values where splitting/ dividing the leaves on could happen
            splits = []
            for loc in range(0, D_count - 1):
                if not iteration[loc][2] == iteration[loc + 1][2]:
                    splits.append((float(iteration[loc][1]) + float(iteration[loc + 1][1])) / 2)
            hypo_flg = 1
            if not splits:
                hypo_flg = 0
        #   Now we have to find which split point is best to use through calculating info gain
        #   We need a list of all of the info gains: info
        info = []
        for split in splits:
            leftp = 0
            leftm = 0
            rightp = 0
            rightm = 0
            for val in iteration:
                if float(val[1]) < float(split):
                    if val[2] == '-':
                        leftm += 1
                    else:
                        leftp += 1
                else:
                    if val[2] == '-':
                        rightm += 1
                    else:
                        rightp += 1
            leadl = (leftp + leftm) / D_count
            leadr = (rightp + rightm) / D_count
            fraclm = leftm / (leftm + leftp)
            fraclp = leftp / (leftm + leftp)
            fracrm = rightm / (rightm + rightp)
            fracrp = rightp / (rightp + rightm)

            if fraclm == 0 and fraclp == 0:
                followl = 0
            elif fraclm == 0 and not fraclp == 0:
                followl = (-1) * (fraclp * math.log(fraclp))
            elif fraclp == 0 and not fraclm == 0:
                followl = (-1) * fraclm * math.log(fraclm)
            else:
                followl = ((-1) * fraclm * math.log(fraclm)) - (fraclp * math.log(fraclp))

            if fracrm == 0 and fracrp == 0:
                followr = 0
            elif fracrm == 0 and not fracrp == 0:
                followr = (-1) * fracrp * math.log(fracrp)
            elif fracrp == 0 and not fracrm == 0:
                followr = (-1) * fracrm * math.log(fracrm)
            else:
                followr = ((-1) * fracrm * math.log(fracrm)) - (fracrp * math.log(fracrp))

            position1 = (leadl * followl) + (leadr * followr)
            position2 = ''
            position3 = ''

            if leftm > leftp:
                position2 = '-'
            else:
                position2 = '+'
            if rightm > rightp:
                position3 = '-'
            else:
                position3 = '+'

            info.append([position1, position2, position3, split])

        #   Now we need to choose the split with the smallest
        splitloc = 0
        splitcount = 0
        splitval = 0
        for val in info:
            if float(val[0]) > splitval:
                splitval = float(val[0])
                splitloc = splitcount
            splitcount += 1

        #   We now have to save the model for output later
        h[i].append(info[splitloc][1])
        h[i].append(info[splitloc][3])
        h[i].append(info[splitloc][2])
        #   We now have the h for this iteration. split D on h
        for val in iteration:
            temp = []
            temp.append(val[1])
            if float(val[1]) < float(info[splitloc][0]):
                temp.append(info[splitloc][1])
            else:
                temp.append(info[splitloc][2])
            D[i + 1].append(temp)

        #   We now need to find correct class on D vs incorrect class on D
        correctpos = []
        for val in D[0]:
            if float(val[0]) < float(h[i][1]):
                if val[1] == h[i][0]:
                    correctpos.append('C')
                else:
                    correctpos.append('I')
            else:
                if val[1] == h[i][2]:
                    correctpos.append('C')
                else:
                    correctpos.append('I')

        #   Now need to calculate Error
        error = 0
        for spot in range(0, D_count):
            if correctpos[spot] == 'I':
                error += W[i][spot]
        E[i].append(error)
        # If the error is 0.5 we have to restart the whole process
        if float(E[i][0]) >= 0.5:
            error_flg = 0
            E[i][0] = 0.6
        else:
            error_flg = 1
        #   Now calculate the new weights
        for spot in range(0, D_count):
            if correctpos[spot] == 'I':
                W[i + 1].append((float(W[i][spot]) / (2 * float(E[i][0]))))
            else:
                W[i + 1].append((float(W[i][spot]) / (2 * (1 - float(E[i][0])))))
        #   Now we find the alpha
        A[i].append(0.5 * math.log(((1 - float(E[i][0])) / float(E[i][0]))))

        if error_flg == 0:
            D[i + 1] = []
            E[i] = []
            A[i] = []
            W[i + 1] = []
            h[i] = []

        else:
            i += 1
with open("outputrounds.txt", "w") as f:
    f.write(f"iteration:\tStart\n")
    f.write(f"\tD:\t{D[0]}\n")

    for spot in range(1, n + 1):
        f.write(f"\niteration:\t{spot}\n")
        f.write(f"\tD:\t{D[spot]}\n")
        f.write(f"\th([left leaf sign, split value, right leaf sign]:\t{h[spot - 1]}\n")
        f.write(f"\tError:\t{E[spot - 1]}\n")
        f.write(f"\tAlpha:\t{A[spot - 1]}\n")
        f.write(f"\tWeight:\t{W[spot - 1]}\n")

#   This is for the final part of the project
#   I'm using a csv file of the input qustion

test = input("Would you like to test a stump? (input y to confirm)\n")
if test == "y":
    title = []
    rows = []
    inputfile = input("Enter csv file of input\n")
    with open(inputfile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                for item in row:
                    title.append(item)
            else:
                rows.append(row)
                line_count += 1

#   Now we use our newly recorded input to find the total number of rows
row_count = 0
for row in rows:
    row_count += 1

#   Now we use our newly recorded input to find the total number of columns
column_count = 0
for column in title:
    column_count += 1

# since we already know that there is only one column I'm going to be lazy here
classification = []
h.pop()
for value in rows:
    total = 0

    position = 0
    for tree in h:
        if float(value[0]) < float(tree[1]):
            if tree[0] == '+':
                total += float(A[position][0])
            else:
                total += float(A[position][0]) * (-1)
        else:
            if tree[2] == '+':
                total += float(A[position][0])
            else:
                total += float(A[position][0]) * (-1)
        position += 1
    if total >= 0:
        classification.append([value[0], '+'])
    else:
        classification.append([value[0], "-"])

with open("outputtestp2.txt", "w") as f:
    f.write(f"x\tClassification\n")
    for item in classification:
        f.write(f"{item[0]}\t{item[1]}\n")

