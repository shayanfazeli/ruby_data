import numpy
import torch
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
from ruby_data.utilities.polygon import get_centroid_and_tlbr
from ruby_data.utilities.wsi import get_whole_slide_filepaths, load_whole_slides_in_memory, build_item_identifiers


class RubyDataset(Dataset):
    def __init__(self, root_path: str = './data'):
        print("initializing...")
        self.whole_slides = get_whole_slide_filepaths(root_path)
        self.fetched_whole_slides = load_whole_slides_in_memory(self.whole_slides)
        self.item_identifiers = build_item_identifiers(self.fetched_whole_slides)

    def __getitem__(self, idx):
        category, image_idx, annotation_idx = self.item_identifiers[idx]
        meta = self.fetched_whole_slides[category][image_idx][1]['annotations'][annotation_idx]
        cell_label, polygon_data = meta['name'], meta['polygon']['paths'][0]

        centroid, (top_left, bottom_right) = get_centroid_and_tlbr(polygon_data, self.fetched_whole_slides[category][image_idx][0].shape[1],
            self.fetched_whole_slides[category][image_idx][0].shape[0], )

        sampled_image = self.fetched_whole_slides[category][image_idx][0][top_left[1]:bottom_right[1],
                        top_left[0]:bottom_right[0], :]

        return sampled_image, cell_label, polygon_data

    def visualize_sample_item(self):
        category, image_idx, annotation_idx = self.item_identifiers[numpy.random.randint(0, len(self.item_identifiers))]

        meta = self.fetched_whole_slides[category][image_idx][1]['annotations'][annotation_idx]
        cell_label, polygon_data = meta['name'], meta['polygon']['paths'][0]

        centroid, (top_left, bottom_right) = get_centroid_and_tlbr(polygon_data, self.fetched_whole_slides[category][image_idx][0].shape[1],
            self.fetched_whole_slides[category][image_idx][0].shape[0], )

        x = [point['x'] - top_left[0] for point in polygon_data]
        y = [point['y'] - top_left[1] for point in polygon_data]

        # To close the polygon, append the first point to the end of the list
        x.append(polygon_data[0]['x'] - top_left[0])
        y.append(polygon_data[0]['y'] - top_left[1])

        # Plot
        fig, ax = plt.subplots(1, 2)
        sampled_image = self.fetched_whole_slides[category][image_idx][0][top_left[1]:bottom_right[1],
                        top_left[0]:bottom_right[0], :]
        ax[0].imshow(sampled_image)
        ax[0].plot(x, y, '-o', color='red')  # '-o' means it will be a line with markers at each point
        ax[0].plot(centroid[0] - top_left[0], centroid[1] - top_left[1], 'x')
        ax[0].fill(x, y, alpha=0.1, color='red')  # Optionally fill the polygon
        ax[0].set_title(f'{cell_label}: {sampled_image.shape[:-1]}')
        ax[0].axis('off')

        ax[1].imshow(self.fetched_whole_slides[category][image_idx][0][top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :])
        ax[1].axis('off')
        ax[1].set_title(f'{sampled_image.shape[:-1]}')

        plt.show()

    def __len__(self):
        return len(self.item_identifiers)


def custom_collate_fn(batch_elements):
    images = torch.stack([torch.from_numpy(e[0]) for e in batch_elements])
    labels = [e[1] for e in batch_elements]
    polygons = [e[2] for e in batch_elements]
    return dict(images=images, labels=labels, polygons=polygons)
