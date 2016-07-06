# coding: utf-8
import zipfile


def img2vector(file_name):
    pass


def run():
    z = zipfile.ZipFile("./bookdemo/Ch02/digits.zip")
    for f in z.namelist():
        print z.read(f)
    test_vector = img2vector()


if __name__=="__main__":
    run()