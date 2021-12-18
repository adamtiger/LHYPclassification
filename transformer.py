from heartcontour.data_wrangling.lvhyp_reader import Patient, EndPhase
from heartcontour.utils import progress_bar
from heartcontour.utils import get_logger
from standard import ViewPlane
from collections import defaultdict
import pickle
import os

logger = get_logger(__name__)


class Sample:

    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.pathology = None
        self.gender = None
        # sa images
        self.sa_bas = None  # list of images
        self.sa_mid = None
        self.sa_api = None
        # la images
        self.la_2ch = None  # list of images
        self.la_4ch = None
        self.la_lvot = None
        # late enhancement images
        self.sale = None
        self.lale = None

    def serialize(self, path):
        with open(path, 'wb') as pc:
            pickle.dump(self, pc)

    @staticmethod
    def deserialize(path):
        with open(path, 'rb') as pc:
            return pickle.load(pc)


class Transform:
    """
    Filtering out some short-axis and long-axis images.
    12 x 3 images from SA:
        - basal slice (every second image)
        - mid slice (every second image)
        - apical slice (every second image)
    12 x 3 images from LA:
        - 2 channel view (every second image)
        - 4 channel view (every second image)
        - 3 channel view (lvot) (every second image)
    3 images from LALE:
        - 1 from each view with contrast enhancement
    6 images from SALE:
        - 6 images from the center part of the series
        - sale is simalr like TRA
    Creates new objects for the training.
    """
    def __init__(self, db_path, source, target):
        self.db_path = db_path
        self.source = source
        self.target = target

    def __filter_la(self, la_images):
        filtered = defaultdict(list)
        for idx, img in enumerate(la_images, 0):
            vp = ViewPlane(*(img.orientation))
            view = vp.calculate_view()
            if idx % 2 == 0:
                if view.name == 'LA2CH':
                    im = img.get_image(self.db_path)
                    filtered['ch2'].append(vp.standardize_image(im))
                elif view.name == 'LA4CH':
                    im = img.get_image(self.db_path)
                    filtered['ch4'].append(vp.standardize_image(im))
                elif view.name == 'LALVOT':
                    im = img.get_image(self.db_path)
                    filtered['lvot'].append(vp.standardize_image(im))
        return filtered

    def __filter_sa(self, sa_images):
        sorted_sas = sorted(sa_images, key=lambda img: img.frame)
        num_slices = len(sa_images) // 25
        bas_idx = num_slices // 3 * 2
        mid_idx = num_slices // 2
        api_idx = num_slices // 3
        filtered = defaultdict(list)
        for img in sorted_sas:
            vp = ViewPlane(*img.orientation)
            if img.frame % 2 == 0:
                if img.slice == bas_idx:
                    im = img.get_image(self.db_path)
                    filtered['bas'].append(vp.standardize_image(im))
                elif img.slice == mid_idx:
                    im = img.get_image(self.db_path)
                    filtered['mid'].append(vp.standardize_image(im))
                elif img.slice == api_idx:
                    im = img.get_image(self.db_path)
                    filtered['api'].append(vp.standardize_image(im))
        return filtered
    
    def __filter_lale(self, lale_images):
        filtered = defaultdict(list)
        for _, img in enumerate(lale_images, 0):
            im = img.get_image(self.db_path)
            filtered['lale'].append(im)  # can be more sofisticated if needed
        return filtered
    
    def __filter_sale(self, sale_images):
        filtered = defaultdict(list)
        mid_idx = len(sale_images) // 2  # six images are needed
        lowest = mid_idx - 3
        highest = mid_idx + 2
        for idx, img in enumerate(sale_images, 0):
            if lowest <= idx <= highest:
                im = img.get_image(self.db_path)
                filtered['sale'].append(im)  # can be more sofisticated if needed
        return filtered

    def __create_sample(self, patient):
        smp = Sample(patient.id)
        smp.pathology = patient.pathology
        smp.gender = patient.clinical_data.gender
        # sa images
        sa_images = self.__filter_sa(patient.sa_images)
        smp.sa_bas = sa_images['bas']
        smp.sa_mid = sa_images['mid']
        smp.sa_api = sa_images['api']
        # la images
        la_images = self.__filter_la(patient.la_images)
        smp.la_2ch = la_images['ch2']
        smp.la_4ch = la_images['ch4']
        smp.la_lvot = la_images['lvot']
        # lale images
        lale_images = self.__filter_lale(patient.lale_images)
        smp.lale = lale_images['lale']
        # sale images
        sale_images = self.__filter_sale(patient.sale_images)
        smp.sale = sale_images['sale']
        return smp

    def __issampletotal(self, sample):
        pathology_exists = not (sample.pathology is None)
        sa_bas = (len(sample.sa_bas) > 0)
        sa_mid = (len(sample.sa_mid) > 0)
        sa_api = (len(sample.sa_api) > 0)
        la_2ch = (len(sample.la_2ch) > 0)
        la_4ch = (len(sample.la_4ch) > 0)
        la_lvot = (len(sample.la_lvot) > 0)
        lale = (len(sample.lale) > 0)
        sale = (len(sample.sale) > 0)
        return pathology_exists and (
            (sa_bas and sa_mid and sa_api) or la_2ch or la_4ch or la_lvot or lale or sale
        )

    def transform_all(self):
        logger.info('Start processing the raw dataset')
        pickles = os.listdir(self.source)
        for cntr, pckl in enumerate(pickles, 1):
            if os.path.exists(os.path.join(self.target, pckl)):
                continue
            path = os.path.join(self.source, pckl)
            patient = Patient.deserialize(path)
            try:
                sample = self.__create_sample(patient)
            except TypeError:
                continue
            except IndexError:
                continue
            if self.__issampletotal(sample):
                path = os.path.join(self.target, sample.patient_id + '.pckl')
                sample.serialize(path)
            else:
                print(
                    sample.patient_id,
                    sample.pathology
                )
            progress_bar(cntr, len(pickles), 20)
        logger.info('Process was finished')


if __name__ == '__main__':
    from heartcontour.data_wrangling.lvhyp_reader import *
    trf = Transform(
        r'D:\AI\works\Heart\data\hypertrophy\cleanready',
        r'D:\AI\works\Heart\data\hypertrophy\middle',
        r'D:\AI\works\Heart\data\hypertrophy\transformed'
    )
    trf.transform_all()
