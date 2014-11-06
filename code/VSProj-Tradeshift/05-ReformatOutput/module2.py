import csv
import gzip

inFilePath= "..\\..\\..\\classifier\\Meta\\output_proba_RF_lastCol.csv"
outFilePath = "..\\..\\..\\output_final.csv.gz"


if __name__ == '__main__':

    with open(inFilePath, 'rb') as infile:
        reader = csv.reader(infile)
        reader.next()

        with gzip.open(outFilePath, 'wb') as outfile:
            outfile.write('id_label,pred\n')
            for row in reader:
                id = row[0]
                predictions = row[1:]

                for y_id, pred in enumerate(predictions):
                    outfile.write('%s_y%s,%s\n' % (id, y_id + 1, pred))