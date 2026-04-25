from modelscope.msdatasets import MsDataset
import os
import pandas as pd

MAX_DATA_NUMBER = 500

if not os.path.exists('coco_2014_caption'):
    ds =  MsDataset.load('modelscope/coco_2014_caption', subset_name='coco_2014_caption', split='train')
    print(len(ds))
 
    total = min(MAX_DATA_NUMBER, len(ds))

    os.makedirs('coco_2014_caption', exist_ok=True)

    image_paths = []
    captions = []

    for i in range(total):
        item = ds[i]
        image_id = item['image_id']
        caption = item['caption']
        image = item['image']

        image_path = os.path.abspath(f'coco_2014_caption/{image_id}.jpg')
        image.save(image_path)

        image_paths.append(image_path)
        captions.append(caption)

        if (i + 1) % 50 == 0:
            print(f'Processing {i+1}/{total} images ({(i+1)/total*100:.1f}%)')

    df = pd.DataFrame({
        'image_path': image_paths,
        'caption': captions
    })

    df.to_csv('./coco-2024-dataset.csv', index=False)

else:
    print('Already exist')
    
