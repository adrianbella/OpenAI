import csv


class MyCSVHandler:

    @staticmethod
    def write_csv_file(source, data):

        with open(source, 'a') as f:
            writer = csv.writer(f, delimiter=' ', quotechar='|')
            writer.writerow([data])
            f.close()

    @staticmethod
    def write_cvs_file_title(source, title):
        with open(source, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([title])
            f.close()