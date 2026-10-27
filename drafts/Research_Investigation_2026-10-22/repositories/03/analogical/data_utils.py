import os
import json
from PIL import Image, ImageEnhance


def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:
        if 'sample' in split:
            filepath = os.path.join(anno_dir, split)
            with open(filepath) as f:
                new_data = json.load(f)

        elif "/" not in split:    # the official splits
            if tokenizer == 'bert':
                filepath = os.path.join(anno_dir, '%s_%s_enc.json' % (dataset.upper(), split))
            elif tokenizer == 'xlm':
                filepath = os.path.join(anno_dir, '%s_%s_enc_xlmr.json' % (dataset.upper(), split))
            else:
                raise NotImplementedError('unsupported tokenizer %s' % tokenizer)

            with open(filepath) as f:
                new_data = json.load(f)

            if split == 'val_train_seen':
                new_data = new_data[:50]

        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)
    return data


def load_obj2vps(bbox_file):
    obj2vps = {}
    bbox_data = json.load(open(bbox_file))
    for scanvp, value in bbox_data.items():
        scan, vp = scanvp.split('_')
        for objid, objinfo in value.items():
            if objinfo['visible_pos']:
                obj2vps.setdefault(scan+'_'+objid, [])
                obj2vps[scan+'_'+objid].append(vp)
    return obj2vps


def add_data_to_json(file_path, new_data):
    with open(file_path, 'w') as file:
        json.dump(new_data, file, indent=4)


def process_img(img):
    img_root, img_id = img.split('/')[:-1],img.split('/')[-1][:-4]
    img_path2 = "/".join(img_root+[img_id+".jpg"])
    img_path1 = "/".join(img_root+[str(int(img_id)-1)+".jpg"])
    img_path3 = "/".join(img_root+[str(int(img_id)+1)+".jpg"])

    image1 = Image.open(img_path1)
    image2 = Image.open(img_path2)
    image3 = Image.open(img_path3)

    # Adjust brightness of image1 and image3
    def darken_image(image, factor=0.5):
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    darkened_image1 = darken_image(image1, factor=0.5)  # 50% darker
    darkened_image3 = darken_image(image3, factor=0.5)  # 50% darker

    total_width = darkened_image1.width + image2.width + darkened_image3.width
    max_height = max(darkened_image1.height, image2.height, darkened_image3.height)
    combined_image = Image.new("RGB", (total_width, max_height))

    # Paste the images side by side
    combined_image.paste(darkened_image1, (0, 0))
    combined_image.paste(image2, (darkened_image1.width, 0))
    combined_image.paste(darkened_image3, (darkened_image1.width + image2.width, 0))
    combined_image.save("combined_image.jpg")
    combined_image.show()