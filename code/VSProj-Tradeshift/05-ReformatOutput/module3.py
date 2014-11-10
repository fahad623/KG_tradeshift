import csv
import gzip

inFilePath= "..\\..\\..\\classifier\\Meta\\output_proba3.csv"
outFilePath = "..\\..\\..\\output_proba_final3.csv.gz"


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
                    outfile.write('%d_y%s,%s\n' % (id, y_id + 1, pred))