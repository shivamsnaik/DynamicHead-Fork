def get_coco_samples(input_file_path, output_file_path, num_samples = 10, shuffle=True):
    """
    Function to create a subset of a given COCO json file
    
    Params:
    - input_file_path = Input file full path
    - output_file_path = Output file full path
    - num_samples = Number of images to be included in the subset
    - shuffle = To choose images in order or randomely.
    
    Return Values:
    - Returns the fetched samples in COCO format
    """
    
    try:
        import json
        import random
        from pycocotools.coco import COCO
        import traceback
        
        coco_dict = {}
        
        with open(input_file_path, "r") as file:
            coco = json.load(file)

        # Store the fixed contents of the coco file
        coco_dict["info"] = coco["info"]
        coco_dict["licenses"] = coco["licenses"]
        coco_dict["categories"] = coco["categories"]
        
        coco = COCO(input_file_path)
        
        # Choose specified number of images (Choose randomely if shuffle is True)
        img_ids = [random.randint(1, 50000) if shuffle else index+1 for index in range(num_samples)]
        coco_dict["images"] = coco.loadImgs(ids=img_ids)
        
        ann_ids = coco.getAnnIds(imgIds=img_ids)
        coco_dict["annotations"] = coco.loadAnns(ids=ann_ids)
        
        output_file = open(output_file_path, "w")
        json.dump(coco_dict, output_file, indent=2)
        
        return (coco_dict)
    except e:
        traceback.print_exception()
        
    finally:
        file.close()
        output_file.close()
