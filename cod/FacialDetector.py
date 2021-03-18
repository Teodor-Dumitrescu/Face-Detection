from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog


class FacialDetector:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_model = None

    def get_positive_descriptors(self):
        # in aceasta functie calculam descriptorii pozitivi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor pozitive
        # iar D - dimensiunea descriptorului
        # D = (params.dim_window/params.dim_hog_cell - 1) ^ 2 * params.dim_descriptor_cell (fetele sunt patrate)

        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []
        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul pozitiv numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)

            descriptor_img = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                 cells_per_block=(2, 2))
            positive_descriptors.append(descriptor_img)

            # add the descriptor for the horizontaly flipped image to the dataset
            img_flipped = cv.flip(img, 1)

            descriptor_img_flipped = hog(img_flipped, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                 cells_per_block=(2, 2))
            positive_descriptors.append(descriptor_img_flipped)
            print("Am extras descriptorul pentru imaginea ", i, " cu dimensiunea de ", descriptor_img.shape)

        positive_descriptors = np.array(positive_descriptors)
        print("Dupa ce am extras toti descriptorii din imaginile pozitive obtinem un array cu dimensiunea ",
              positive_descriptors.shape)
        return positive_descriptors

    def get_negative_descriptors(self):
        # in aceasta functie calculam descriptorii negativi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor negative
        # iar D - dimensiunea descriptorului
        # avem 274 de imagini negative, vream sa avem self.params.number_negative_examples (setat implicit cu 10000)
        # de exemple negative, din fiecare imagine vom genera aleator self.params.number_negative_examples // 274
        # patch-uri de dimensiune 36x36 pe care le vom considera exemple negative

        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        num_negative_per_image = self.params.number_negative_examples // num_images
        negative_descriptors = []
        print('Calculam descriptorii pt %d imagini negative' % num_images)
        for i in range(num_images):
            print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)

            H = img.shape[0]  # height of image
            W = img.shape[1]  # width of image

            # (xmin, ymin) is the top-left corner of a window
            # (xmax, ymax) is the bottom-right corner of a window; xmax = xmin + 35
            # ymax = ymin + 35
            xmin = np.random.randint(0, W - self.params.dim_window, num_negative_per_image)
            xmax = xmin + self.params.dim_window

            ymin = np.random.randint(0, H - self.params.dim_window, num_negative_per_image)
            ymax = ymin + self.params.dim_window

            for idx in range(len(xmin)):
                window = img[ymin[idx]: ymax[idx], xmin[idx]: xmax[idx]]

                descriptor_window = hog(window, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                        cells_per_block=(2, 2))
                negative_descriptors.append(descriptor_window)

        negative_descriptors = np.array(negative_descriptors)
        print("Dupa ce am extras toti descriptorii pentru imaginile negative obtinem un array de dimensiuni: ",
              negative_descriptors.shape)

        return negative_descriptors

    def train_classifier(self, training_examples, train_labels):
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        # Cs = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 2, 3]
        Cs = [0.01]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            # model = LinearSVC(random_state=0, C=c, tol=1e-5, max_iter=10000)
            model = LinearSVC(random_state=0, C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]

        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(negative_scores) + 20))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area + 10 ** (-14))

        return iou

    def non_maximum_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]

        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[
                        j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],
                                                        sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False

        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def negative_mining(self):
        test_images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        detections = []  # lista cu toate detectiile pe care le obtinem
        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]
        num_test_images = len(test_files)

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea negativa %d/%d..' % (i, num_test_images))
            img_vanilla = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            H, W = img_vanilla.shape

            # compute the number of blocks on width and height for image
            blocks_height = H // self.params.dim_hog_cell - 1
            blocks_width = W // self.params.dim_hog_cell - 1

            # compute the number of blocks on width/height for a window
            blocks_window = int(self.params.dim_window / self.params.dim_hog_cell - 1)

            # get hog descriptor for img
            img_desc = hog(img_vanilla, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2))

            # reshape the descriptor so we can slice it easier
            img_desc = img_desc.reshape((blocks_height, blocks_width, self.params.dim_descriptor_cell))

            # slide window and get it's descriptor
            for b_h in range(blocks_height - blocks_window + 1):
                for b_w in range(blocks_width - blocks_window + 1):
                    window_desc = img_desc[b_h: b_h + blocks_window, b_w: b_w + blocks_window, :]

                    # flatten the descriptor for the window for the next operations
                    window_desc = window_desc.flatten()

                    # get the score for the window
                    # score = self.best_model.decision_function(window_desc.reshape((1, -1)))
                    score = np.dot(window_desc.reshape((1, window_desc.shape[0])), w) + bias
                    score = score[0][0]

                    # if score > threshold we consider it a detection
                    if score > self.params.threshold_mining:
                        detections.append(window_desc)

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii negative %d/%d este %f sec.'
                  % (i, num_test_images, end_time - start_time))

        detections = np.array(detections)

        print("Ferestre puternic negative: ", detections.shape[0])

        return detections

    def run(self, return_descriptors=False):
        """
        Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din self.params.dir_test_examples
        Directorul cu numele self.params.dir_test_examples contine imagini ce
        pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
        date MIT+CMU dar si pentru alte imagini (imaginile realizate cu voi la curs+laborator).
        Functia 'non_maximum_suppression' suprimeaza detectii care se suprapun (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
        Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.
        :return:
        detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
        detections[i, :] = [x_min, y_min, x_max, y_max]
        scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
        file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
        (doar numele, nu toata calea).
        """
        # scale_options = np.linspace(0.2, 3, 29)
        # scale_options = np.linspace(0.1, 2, 30)
        # scale_options = np.array([1])
        # scale_options = np.concatenate((np.linspace(0.05, 1, 20), np.linspace(1, 2, 5)[1:]), axis=0)
        # am ramas la varianta aceasta deoarece scala > 1 adauga multe detectii false positive si strica mult avg. prec
        scale_options = np.linspace(0.1, 1, 10)

        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        detections = np.array([[0, 0, 0, 0]])  # array cu toate detectiile pe care le obtinem
        scores = np.array([0])  # array cu toate scorurile pe care le optinem
        file_names = []  # array cu fisiele, in aceasta lista fisierele vor aparea de mai multe ori, pentru fiecare
        # detectie din imagine, numele imaginii va aparea in aceasta lista
        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]
        print("w ", w)
        print("bias ", bias)
        num_test_images = len(test_files)
        descriptors_to_return = []
        files = os.listdir(self.params.dir_test_examples)

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            img_vanilla = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)

            # scores and detections for current image
            scores_i = []
            detections_i = []

            # slide window on image at different scales
            for scale in scale_options:
                print("Scara: ", scale)
                
                # transform scores and detections from arrays to list because the append operation is much faster than
                # concatenate
                scores_i = list(scores_i)
                detections_i = list(detections_i)

                # get dimensions of image for current scale
                H = int(img_vanilla.shape[0] * scale)
                W = int(img_vanilla.shape[1] * scale)

                # resize the image to current scale
                img = cv.resize(img_vanilla, (W, H))
                
                if img.shape[0] < self.params.dim_window or img.shape[1] < self.params.dim_window:
                    continue

                # compute the number of blocks on width and height for image
                blocks_height = H // self.params.dim_hog_cell - 1
                blocks_width = W // self.params.dim_hog_cell - 1

                # compute the number of blocks on width/height for a window
                blocks_window = int(self.params.dim_window / self.params.dim_hog_cell - 1)

                # get hog descriptor for img
                img_desc = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                               cells_per_block=(2, 2))

                # reshape the descriptor so we can slice it easier
                img_desc = img_desc.reshape((blocks_height, blocks_width, self.params.dim_descriptor_cell))

                # slide window and get it's descriptor
                for b_h in range(blocks_height - blocks_window + 1):
                    for b_w in range(blocks_width - blocks_window + 1):
                        window_desc = img_desc[b_h: b_h + blocks_window, b_w: b_w + blocks_window, :]

                        # flatten the descriptor for the window for the next operations
                        window_desc = window_desc.flatten()

                        # get the score for the window
                        # score = self.best_model.decision_function(window_desc.reshape((1, -1)))
                        score = np.dot(window_desc.reshape((1, window_desc.shape[0])), w) + bias
                        score = score[0][0]

                        # if score > threshold we consider it a detection
                        if score > self.params.threshold:
                            scores_i.append(score)
                            x_min = b_w * self.params.dim_hog_cell * (1 / scale)
                            y_min = b_h * self.params.dim_hog_cell * (1 / scale)
                            x_max = x_min + self.params.dim_window * (1 / scale)
                            y_max = y_min + self.params.dim_window * (1 / scale)
                            detections_i.append([int(x_min), int(y_min), int(x_max), int(y_max)])

            scores_i = np.array(scores_i)
            detections_i = np.array(detections_i)

            if scores_i.shape[0] == 0:
                continue

            print("inainte de suppression: ", detections_i.shape)
            detections_i, scores_i = self.non_maximum_suppression(detections_i, scores_i, img_vanilla.shape)
            print("dupa suppression: ", detections_i.shape)
            for j in range(scores_i.shape[0]):
                file_names.append(files[i])

            detections = np.concatenate((detections, detections_i), axis=0)
            scores = np.concatenate((scores, scores_i), axis=0)

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
                  % (i, num_test_images, end_time - start_time))

        file_names = np.array(file_names)

        detections = detections[1:]
        scores = scores[1:]

        if return_descriptors:
            return descriptors_to_return
        return detections, scores, file_names

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()
