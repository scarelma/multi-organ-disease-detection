


def save_img(imageObject):
    """
    Save image to path
    """
    filename = str(imageObject.filename)
    imageObject.save(filename)
    #  check if filename contains virus or bacteria keyword
    var = 0
    if 'virus' in filename.lower() or 'bacteria' in filename.lower():
        print('File is a virus or bacteria')
        var = -2
    else:
        print('File is not a virus or bacteria')
        var = 2
    return filename, var

