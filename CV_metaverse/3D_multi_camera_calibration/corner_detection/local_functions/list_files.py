import glob
def list_files(root,fileType):
    '''
    create a list of all files in the root folder which have same fileType 
    
    input:
        root: folder path
        fileType: the file is this case *.jpg/png
    '''
    path=root +  "/*." + fileType         
    files = glob.glob(path)#, recursive=True)     
    return files  
    
if __name__ == '__main__':      
    root = r'/Users/farshid/code/pirahansiah/CV_metaverse/3D_multi_camera_calibration/corner_detection/dataSet'
    filesName=list_files(root,"png")
    print(filesName)
    