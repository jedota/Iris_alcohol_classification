import os
import cv2
import json
import glob
import imgaug
import skimage
import numpy as np
from PIL import Image
from imutils.paths import list_images
from tensorflow.keras.utils import Sequence

# fancy load_image method
def load_image(path, colorspace="RGB"):
    color = colorspace.lower()
    spaces = {
        "rgb": cv2.COLOR_BGR2RGB,
        "hsv": cv2.COLOR_BGR2HSV,
        "hsv_full": cv2.COLOR_BGR2HSV_FULL,
        "gray": cv2.COLOR_BGR2GRAY,
        "lab": cv2.COLOR_BGR2LAB,
    }

    if color not in spaces.keys():
        print(f"[WARNING] color space {colorspace} not supported")
        print(f"Supported list: {spaces.keys()}")
        print("Colorspace setted to RGB")
        color = "rgb"

    image = cv2.cvtColor(cv2.imread(path), spaces[color])
    return image


# base Dataset class which inherets from Sequence class from Keras
# add some abstract functions to be implemented in the child classes
class Dataset(Sequence):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class EyesDataset(Dataset):
            def load_eyes(self):
                    ...
            def load_mask(self, image_id):
                    ...
            def image_reference(self, image_id):
                    ...
    """

    def __init__(
        self,
        shuffle=True,
        dim=(480, 640),
        color_space="RGB",
        augmentation=None,
        channels=3,
        batch_size=1,
        class_map=None,
        preprocess=[],
    ):
        self.dim = dim
        self._image_ids = []
        self.image_info = []
        self.shuffle = shuffle
        self.channels = channels
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.input_shape = (*dim, channels)
        self.class_info = []
        self.preprocess = preprocess
        self.source_class_ids = {}
        self.color_space = color_space

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info["source"] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append(
            {
                "source": source,
                "id": class_id,
                "name": class_name,
            }
        )

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
                  classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)
        np.random.shuffle(self._image_ids)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {
            "{}.{}".format(info["source"], info["id"]): id
            for info, id in zip(self.class_info, self.class_ids)
        }
        self.image_from_source_map = {
            "{}.{}".format(info["source"], info["id"]): id
            for info, id in zip(self.image_info, self.image_ids)
        }

        # Map sources to class_ids they support
        self.sources = list(set([i["source"] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info["source"]:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info["source"] == source
        return info["id"]

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_ids):
        """Load the specified image and return a [BS,H,W,3] Numpy array."""
        bs = np.zeros([self.batch_size, *self.dim, self.channels], np.uint8)

        for i, idx in enumerate(image_ids):
            # Load image
            image = load_image(
                self.image_info[idx]["path"], colorspace=self.color_space
            )
            image = cv2.resize(image, self.dim[::-1])  # fuck you cv2

            # asign image to batch
            bs[
                i,
            ] = image

        return bs

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
                masks: A bool array of shape [height, width, instance count] with
                        a binary mask per instance.
                class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning(
            "You are using the default load_mask(), maybe you need to define your own one."
        )
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle == True:
            np.random.shuffle(self._image_ids)

    def __len__(self):
        raise NotImplementedError("abstract method '__len__' not implemented")

    def __getitem__(self, index):
        raise NotImplementedError("abstract method '__getitem__' not implemented")


class EyeDataset(Dataset):
    def load_eyes(self, dataset_dir, subset):
        """Load a subset of the Eye dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train, test or val
        """
        # Add classes
        self.add_class("eye", 0, "eye")
        self.add_class("eye", 1, "iris")
        self.add_class("eye", 2, "pupil")
        self.add_class("eye", 3, "sclera")
        # self.add_class("eye", 4, "eye")

        # Train, test or validation dataset?
        assert subset in ["train", "test", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        """
		# Load annotations
		# regions = list of regions
		# regions:
		# {
		# 	image_name: [
		# 		{
		# 			'shape_attributes': {
		# 				'name': 'polygon', 
		# 				'all_points_x': [...], 
		# 				'all_points_y': [...]
		# 			}, 
		# 			'region_attributes': {'Eye': 'iris'}
		# 		}
		# 		...
		# 	]
		# }
		"""

        # load all regions.json files
        # each region file contains a list of regions and their attributes
        annotations = json.load(open(os.path.join(dataset_dir, "regions.json")))

        # Add images
        for key in annotations:
            # key = image_name
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            polygons = [r["shape_attributes"] for r in annotations[key]]
            objects = [s["region_attributes"] for s in annotations[key]]

            # num_ids = [1, 2, 3] => ['iris', 'pupil', 'sclera', ]
            num_ids = []

            for obj in objects:
                for cl_info in self.class_info:
                    if cl_info["name"] == obj["Eye"]:
                        num_ids.append(cl_info["id"])

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, key)
            # height, width = skimage.io.imread(image_path).shape[:2]
            # Pillow use less memory
            width, height = Image.open(image_path).size

            self.add_image(
                "eye",
                image_id=key,  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                num_ids=num_ids,
            )

    def load_mask(self, image_ids):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [bs, height, width, instance count] with
                one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # create a reference of batch mask filled with zeros
        bs_mask = np.zeros(
            [self.batch_size, *self.dim, self.num_classes], dtype=np.float32
        )

        for idx, imid in enumerate(image_ids):
            # If not an eye dataset image, delegate to parent class.
            image_info = self.image_info[imid]
            if image_info["source"] != "eye":
                return super(self.__class__, self).load_mask(imid)
            num_ids = image_info["num_ids"]

            # Convert polygons to a bitmap mask of shape
            # [height, width, instance_count]
            info = self.image_info[imid]

            # create a mask reference for each class
            mask = np.zeros(
                [image_info["height"], image_info["width"], self.num_classes],
                dtype=np.bool,
            )

            # for each polygon in the image info
            for i, p in zip(num_ids, image_info["polygons"]):
                # Get indexes of pixels inside the polygon and set them to 1
                # try/except is needed to guard against IndexError and KeyError
                try:
                    # if figure is polygon or polyline
                    # use draw.polygon from skimage.draw
                    if p["name"] in [
                        "polygon",
                        "polyline",
                    ]:
                        rr, cc = skimage.draw.polygon(
                            p["all_points_y"], p["all_points_x"]
                        )

                    # do the same for ellipse figure
                    elif p["name"] in [
                        "ellipse",
                    ]:
                        rr, cc = skimage.draw.ellipse(
                            p["cy"], p["cx"], p["ry"], p["rx"]
                        )

                    # do the same for circle figure
                    elif p["name"] in [
                        "circle",
                    ]:
                        rr, cc = skimage.draw.circle(p["cy"], p["cx"], p["r"])

                    # then fill the mask with True values
                    mask[rr, cc, i] = True

                except (KeyError, IndexError) as ex:
                    print(image_info, i, p, ex)

            # fix to iris without pupil

            # locate id of each figure
            id_pupil = next(d["id"] for d in self.class_info if "pupil" in d["name"])
            id_iris = next(d["id"] for d in self.class_info if "iris" in d["name"])
            id_eye = next(d["id"] for d in self.class_info if "eye" in d["name"])

            # get a copy of the mask of pupil and iris
            pupil = mask[..., id_pupil].copy().astype(np.bool)
            iris = mask[..., id_iris].copy().astype(np.bool)

            # do logical xor to get only the region of iris without the pupil
            # iris will be like a donut with a hole in the middle
            iris_xor = np.logical_xor(iris, pupil)
            iris_xor = np.where(iris_xor == True, 1.0, 0.0)

            # replace the iris with the iris_xor
            mask[..., id_iris] = iris_xor.copy()

            # join all pixels generated to get the mask of the eye
            # eye = background
            # eye = not iris + not pupil + not sclera
            one_mask = np.argmax(mask, axis=-1)
            mask[..., id_eye] = np.where(one_mask == id_eye, 1.0, 0.0)

            # resize mask to the same size of the image
            mask = cv2.resize(mask.astype(np.uint8), self.dim[::-1])
            mask = mask.astype(np.bool)

            # assign the mask to the batch mask
            bs_mask[
                idx,
            ] = mask.astype(np.float32)

        return bs_mask

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "eye":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def __len__(self):
        "Denotes the number of batches per epoch"
        return self.num_images // self.batch_size

    def dataAugmentation(self, image, masks):
        # This requires the imgaug lib (https://github.com/aleju/imgaug)
        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = [
            "KeepSizeByResize",
            "CropToFixedSize",
            "TranslateX",
            "TranslateY",
            "Pad",
            "Lambda",
            "Sequential",
            "SomeOf",
            "OneOf",
            "Sometimes",
            "Affine",
            "PiecewiseAffine",
            "CoarseDropout",
            "Fliplr",
            "Flipud",
            "CropAndPad",
            "PerspectiveTransform",
        ]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        for bs in range(self.batch_size):
            # Store shapes before augmentation to compare
            image_shape = image[
                bs,
            ].shape
            mask_shape = masks[
                bs,
            ].shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = self.augmentation.to_deterministic()
            image[bs,] = det.augment_image(
                image[
                    bs,
                ]
            )

            # for each mask, slow?, dunno
            # for c in range(masks[bs, ].shape[-1]):
            # 	uint8_mask = masks[bs, ..., c].astype(np.uint8)

            # 	masks[bs, ..., c] = det.augment_image(
            # 		uint8_mask, hooks=imgaug.HooksImages(activator=hook)
            # 	).astype(np.float32)

            # apply augmentation to masks in one shot
            masks[bs, ...] = det.augment_image(
                masks[bs, ...].astype(np.uint8),
                hooks=imgaug.HooksImages(activator=hook),
            ).astype(np.float32)

            # assert that shapes didn't change
            assert (
                image[
                    bs,
                ].shape
                == image_shape
            ), "Augmentation shouldn't change image size"
            assert (
                masks[
                    bs,
                ].shape
                == mask_shape
            ), "Augmentation shouldn't change mask size"

        return image, masks

    def __getitem__(self, index):
        if index > self._image_ids.max():
            raise IndexError(
                f"List index out of range. Size of generator: {self.__len__()}"
            )
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self._image_ids[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        # load images and masks for each index
        images = self.load_image(indexes)
        masks = self.load_mask(indexes)

        # apply preprocessing step only to images
        # do not apply to masks because its not handled
        if self.preprocess:
            for func in self.preprocess:
                for bs in range(self.batch_size):
                    images[bs, ...] = func(images[bs, ...])

        # data augmentation for images and masks
        # it can be applied to masks as well because imgaug is safe
        if self.augmentation:
            images, masks = self.dataAugmentation(images, masks)

        # normalize images
        images = images / 255.0

        return images, masks

    # get image and mask by image name from the dataset
    def get_image_by_name(self, imname):
        assert self.batch_size == 1, "Batch size must be 1."
        for i in range(len(self._image_ids)):
            if imname in self.image_reference(i):
                return self.__getitem__(i)
