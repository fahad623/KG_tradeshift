import csv

inFilePath= "..\\..\\..\\classifier\\RandomForest\\output_proba.csv"
outFilePath = "..\\..\\..\\output_final.csv"


if __name__ == '__main__':

    with open(inFilePath, 'rb') as infile:
        reader = csv.reader(infile)
        reader.next()

        with open(outFilePath, 'w') as outfile:
            outfile.write('id_label,pred\n')
            for row in reader:
                id = row[0]
                predictions = row[1:]

                for y_id, pred in enumerate(predictions):
                    outfile.write('%s_y%s,%s\n' % (id, y_id + 1, pred))