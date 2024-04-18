import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button, RadioButtons
from tkinter import filedialog

import os
import numpy as np
from PIL import Image

from pydicom import dcmread
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage


class MetaData:
    def __init__(self, meta):
        self.filter = 0
        self.patientName = ""
        self.studyDate = ""
        self.imageComments = ""
        self.patientID = ""
        attribute_mapping = {
            "patientName": "PatientName",
            "studyDate": "StudyDate",
            "imageComments": "ImageComments",
            "patientID": "PatientID"
        }
        for attribute, key in attribute_mapping.items():
            setattr(self, attribute, meta.get(key, ""))


def load_file(image_path: str, res: float = 1.0) -> tuple[dict, np.ndarray]:
    _, extension = os.path.splitext(image_path)

    if extension.lower() == '.dcm':
        ds = dcmread(image_path)
        meta = {attr: getattr(ds, attr) for attr in dir(ds)
                if not callable(getattr(ds, attr))}
        img = Image.fromarray(ds.pixel_array.astype(np.uint8))
    else:
        meta = {}
        img = Image.open(image_path)

    img = img.convert('L')
    new_size = int(max(img.size) * res)
    img.thumbnail((int(res * img.size[0]), int(res * img.size[1])))
    new_img = Image.new('L', (new_size, new_size), 0)
    offset = ((new_size - img.size[0]) // 2, (new_size - img.size[1]) // 2)
    new_img.paste(img, offset)

    return meta, np.array(new_img.getdata()).reshape(new_img.size[0], new_img.size[1])


def calculate_emitter_coords(alpha0: float, alpha1: float, radius: float) -> tuple[float, float]:
    return radius * np.cos(alpha0 + alpha1) + radius, radius * np.sin(alpha0 + alpha1) + radius


def calculate_receiver_coords(alpha0: float, alpha1: float, radius: float) -> tuple[float, float]:
    return radius * np.cos(alpha0 + np.pi - alpha1) + radius, radius * np.sin(alpha0 + np.pi - alpha1) + radius


def bresenham_algorithm(e_point: tuple, r_point: tuple) -> list[tuple[float, float]]:
    x1, y1 = map(int, e_point)
    x2, y2 = map(int, r_point)
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steep = dy > dx
    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        dx, dy = dy, dx
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    error = int(dx / 2)
    y = y1
    y_step = 1 if y1 < y2 else -1
    for x in range(x1, x2 + 1):
        if steep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= dy
        if error < 0:
            y += y_step
            error += dx
    return points


def generate_kernel(size: int) -> np.array:
    if size % 2 == 0 or size < 1:
        return np.array([1])
    middle = size // 2
    filter_kernel = np.ones(size)
    for k in range(1, size // 2 + 1):
        if k % 2 == 0:
            filter_kernel[middle - k] = filter_kernel[middle + k] = 0
        else:
            filter_kernel[middle - k] = filter_kernel[middle + k] = (-4 / (np.pi ** 2)) / (k ** 2)
    return filter_kernel


def convolution_filter(sinogram: np.ndarray, filter_kernel: np.array) -> np.array:
    filtered_sinogram = np.zeros_like(sinogram)
    for i, row in enumerate(sinogram):
        filtered_row = np.convolve(sinogram[i, :], filter_kernel, mode='same')
        filtered_sinogram[i, :] = filtered_row
    return filtered_sinogram


def generate_sinogram(img_path: str, img_res=1.0, emitters_number=180, emitters_range=180,
                      angle_step=1.0, angle_range=180, kernel_size=21) -> tuple:
    print("Generating sinogram")
    _, image = load_file(img_path, img_res)
    img_size = image.shape[0]
    # emitters_number = int(min(emitters_res, 1) * img_size)
    offset_angles = np.deg2rad(np.linspace(-emitters_range/2, emitters_range/2, emitters_number))
    main_angles = np.deg2rad(np.arange(0, angle_range, angle_step))
    angles_num = len(main_angles)
    kernel = generate_kernel(kernel_size)

    sinograms = [np.zeros((angles_num, emitters_number)) for _ in range(angles_num)]
    sinograms_filtered = [np.zeros((angles_num, emitters_number)) for _ in range(angles_num)]
    for iteration, main_angle in enumerate(main_angles):
        sinogram_iteration = []
        print(".", end='')
        for offset_angle in offset_angles:
            emitter_coords = (calculate_emitter_coords
                              (main_angle, offset_angle, img_size / 2))
            receiver_coords = (calculate_receiver_coords
                               (main_angle, offset_angle, img_size / 2))
            beam_points = bresenham_algorithm(emitter_coords, receiver_coords)
            color = []
            for x, y in beam_points:
                if x < img_size and y < img_size:
                    color.append(image[x, y])
            sinogram_iteration.append(np.mean(color) if color else 0)
        sinograms[iteration][iteration] += sinogram_iteration
        sinograms_filtered[iteration] = convolution_filter(sinograms[iteration], kernel)
        if iteration < angles_num - 1:
            sinograms[iteration + 1] += sinograms[iteration]
    print()
    return sinograms, sinograms_filtered


def inverse_sinogram(sinogram: np.array, emitters_number=180, emitters_range=180,
                     angle_step=1.0, angle_range=180) -> list:
    print("Inversing sinogram")
    img_size = len(sinogram[0])
    # emitters_number = int(min(emitters_res, 1) * img_size)
    offset_angles = np.deg2rad(np.linspace(-emitters_range/2, emitters_range/2, emitters_number))
    main_angles = np.deg2rad(np.arange(0, angle_range, angle_step))
    angles_num = len(main_angles)
    images = [np.zeros((img_size, img_size)) for _ in range(angles_num)]

    for x_sinogram, main_angle in enumerate(main_angles):
        print(".", end='')
        for y_sinogram, offset_angle in enumerate(offset_angles):
            emitter_coords = calculate_emitter_coords(main_angle, offset_angle, img_size / 2)
            receiver_coords = calculate_receiver_coords(main_angle, offset_angle, img_size / 2)
            beam_points = bresenham_algorithm(emitter_coords, receiver_coords)
            color = sinogram[x_sinogram][y_sinogram]
            for x, y in beam_points:
                if x < img_size and y < img_size:
                    images[x_sinogram][x, y] += color
        if x_sinogram < angles_num - 1:
            images[x_sinogram + 1] += images[x_sinogram]
    print()
    for i in range(len(images)):
        min_val = np.min(images[i][images[i] != 0])
        max_val = np.max(images[i])
        images[i][images[i] == 0] = min_val
        images[i] = ((images[i] - min_val) / (max_val - min_val)) * 255
    return images


def save_dicom(img: np.ndarray, meta: MetaData) -> None:
    file_path = filedialog.asksaveasfilename(
        filetypes=[("DICOM files", "*.dcm")], defaultextension=".dcm")
    if not file_path:
        return

    img_min = np.min(img)
    img_max = np.max(img)
    out_min, out_max = (0.0, 1.0)
    img_converted = (((img - img_min) / (img_max - img_min) *
                      (out_max - out_min) + out_min) * 255).astype(np.uint8)

    # Create FileMetaDataset
    fmd = FileMetaDataset()
    fmd.MediaStorageSOPClassUID = CTImageStorage
    fmd.MediaStorageSOPInstanceUID = generate_uid()
    fmd.TransferSyntaxUID = ExplicitVRLittleEndian

    # Create Dataset
    ds = Dataset()
    ds.PatientName = meta.patientName
    ds.PatientID = meta.patientID
    ds.ImageComments = meta.imageComments

    ds.StudyDate = meta.studyDate
    ds.file_meta = fmd
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = fmd.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.SamplesPerPixel = 1
    ds.HighBit = 7
    ds.ImagesInAcquisition = 1
    ds.InstanceNumber = 1
    ds.Rows, ds.Columns = img_converted.shape
    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.PixelData = img_converted.tobytes()

    # Save the DICOM file
    ds.save_as(file_path, write_like_original=False)
    print("Saved")
    plt.close()


def rmse(image1: np.ndarray, image2: np.ndarray) -> float:
    if image1.shape[0] >= image2.shape[0]:
        pillow_image = Image.fromarray(image1)
        pillow_image.thumbnail(image2.shape)
        image1 = np.array(pillow_image.getdata()).reshape(pillow_image.size[0], pillow_image.size[1])
    else:
        pillow_image = Image.fromarray(image2)
        pillow_image.thumbnail(image1.shape)
        image2 = np.array(pillow_image.getdata()).reshape(pillow_image.size[0], pillow_image.size[1])
    return np.sqrt(np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)) / 255


def rmse_test() -> None:
    test_fig = plt.figure(figsize=(10, 4))
    test_ax1 = plt.subplot2grid((1, 3), (0, 0))
    test_ax2 = plt.subplot2grid((1, 3), (0, 1))
    test_ax3 = plt.subplot2grid((1, 3), (0, 2))
    fn = "data/Shepp_logan.jpg"
    res = 0.25
    _, og_im = load_file(fn, res)

    test_ax1.set_xlabel('Number of detectors')
    test_ax1.set_ylabel('RMSE value')
    emitters_y = []
    emitters_y_fil = []
    emitters_x = range(90, 721, 90)
    for e_num in emitters_x:
        sin = generate_sinogram(fn, res, e_num)
        inv = inverse_sinogram(sin[0][-1], e_num)
        emitters_y.append(rmse(np.copy(og_im), inv[-1]))
        inv = inverse_sinogram(sin[1][-1], e_num)
        emitters_y_fil.append(rmse(np.copy(og_im), inv[-1]))

    test_ax1.plot(emitters_x, emitters_y, label='Filter off')
    test_ax1.plot(emitters_x, emitters_y_fil, label='Filter on')
    test_ax1.legend()

    test_ax2.set_xlabel('Number of scans')
    test_ax2.set_ylabel('RMSE value')
    scans_y = []
    scans_y_fil = []
    scans_x = range(90, 721, 90)
    for scans_num in scans_x:
        sin = generate_sinogram(fn, res, angle_step=180/scans_num)
        inv = inverse_sinogram(sin[0][-1], angle_step=180/scans_num)
        scans_y.append(rmse(np.copy(og_im), inv[-1]))
        inv = inverse_sinogram(sin[1][-1], angle_step=180/scans_num)
        scans_y_fil.append(rmse(np.copy(og_im), inv[-1]))
    test_ax2.plot(scans_x, scans_y, label='Filter off')
    test_ax2.plot(scans_x, scans_y_fil, label='Filter on')
    test_ax2.legend()

    test_ax3.set_xlabel('Angle range')
    test_ax3.set_ylabel('RMSE value')
    angle_range_y = []
    angle_range_y_fil = []
    angle_range_x = range(45, 270, 45)
    for angle_range in angle_range_x:
        sin = generate_sinogram(fn, res, angle_step=angle_range/180, angle_range=angle_range)
        inv = inverse_sinogram(sin[0][-1], angle_step=angle_range/180, angle_range=angle_range)
        angle_range_y.append(rmse(np.copy(og_im), inv[-1]))
        inv = inverse_sinogram(sin[1][-1], angle_step=angle_range/180, angle_range=angle_range)
        angle_range_y_fil.append(rmse(np.copy(og_im), inv[-1]))
    test_ax3.plot(angle_range_x, angle_range_y, label='Filter off')
    test_ax3.plot(angle_range_x, angle_range_y_fil, label='Filter on')
    test_ax3.legend()

    test_fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # *** PARAMETERS ***
    fig = plt.figure(figsize=(12, 7))
    my_img_res = 0.25
    my_emitters_number = 256
    my_emitters_range = 180
    my_angle_range = 180
    my_angle_step = 1
    my_kernel_size = 21
    my_file_name = filedialog.askopenfilename(filetypes=[("JPG and DICOM files", "*.jpg;*.jpg;*.dcm"),
                                                         ("JPG files", "*.jpg"), ("DICOM files", "*.dcm")])

    # *** PLOTS ***
    ax1 = plt.subplot2grid((2, 3), (0, 0))
    ax2 = plt.subplot2grid((2, 3), (0, 1))
    ax3 = plt.subplot2grid((2, 3), (0, 2))

    ax1.set_title("Original")
    loaded_meta, image_original = load_file(my_file_name, my_img_res)
    new_meta = MetaData(loaded_meta)
    ax1.imshow(image_original, cmap='gray', aspect='auto')

    ax2.set_title("Sinogram")
    images_sinogram = generate_sinogram(my_file_name, my_img_res, my_emitters_number,
                                        my_emitters_range, my_angle_step, my_angle_range, my_kernel_size)
    ax2.imshow(images_sinogram[new_meta.filter][-1], cmap='gray', aspect='auto')

    ax3.set_title("Inversed")
    inversed_sinograms = (inverse_sinogram(images_sinogram[0][-1], my_emitters_number, my_emitters_range,
                                           my_angle_step, my_angle_range),
                          inverse_sinogram(images_sinogram[1][-1], my_emitters_number, my_emitters_range,
                                           my_angle_step, my_angle_range))
    ax3.imshow(inversed_sinograms[new_meta.filter][-1], cmap='gray', aspect='auto')

    # *** WIDGETS ***
    ax_slider_inv = plt.axes((0.59, 0.26, 0.36, 0.03))
    slider_inv = Slider(ax_slider_inv, 'Inv iter', 0, my_angle_range - 1,
                        valinit=my_angle_range - 1, valstep=my_angle_step)
    slider_inv.on_changed(lambda val: ax3.imshow(inversed_sinograms[new_meta.filter][int(val//my_angle_step)],
                                                 cmap='gray', aspect='auto'))

    ax_slider_sin = plt.axes((0.59, 0.36, 0.36, 0.03))
    slider_sin = Slider(ax_slider_sin, 'Sin iter', 0, my_angle_range - 1,
                        valinit=my_angle_range - 1, valstep=my_angle_step)
    slider_sin.on_changed(lambda val: ax2.imshow(images_sinogram[new_meta.filter][int(val//my_angle_step)],
                                                 cmap='gray', aspect='auto'))

    def radio_handler(val):
        if val == 'filter off':
            new_meta.filter = 0
            ax2.imshow(images_sinogram[0][-1], cmap='gray', aspect='auto')
            ax3.imshow(inversed_sinograms[0][-1], cmap='gray', aspect='auto')
        else:
            new_meta.filter = 1
            ax2.imshow(images_sinogram[1][-1], cmap='gray', aspect='auto')
            ax3.imshow(inversed_sinograms[1][-1], cmap='gray', aspect='auto')
        slider_sin.set_val(slider_sin.valmax)
        slider_inv.set_val(slider_inv.valmax)

    ax_radio_filter = plt.axes((0.59, 0.05, 0.13, 0.1))
    radio_filter = RadioButtons(ax_radio_filter, ('filter off', 'filter on'))
    radio_filter.on_clicked(radio_handler)

    name_ax = plt.axes((0.09, 0.35, 0.4, 0.05))
    name_box = TextBox(name_ax, 'Name:', initial=new_meta.patientName)
    name_box.on_submit(lambda text: setattr(new_meta, 'patientName', text))

    id_ax = plt.axes((0.09, 0.25, 0.4, 0.05))
    id_box = TextBox(id_ax, 'ID:', initial=new_meta.patientID)
    id_box.on_submit(lambda text: setattr(new_meta, 'patientID', text))

    date_ax = plt.axes((0.09, 0.15, 0.4, 0.05))
    date_box = TextBox(date_ax, 'Date:', initial=new_meta.studyDate)
    date_box.on_submit(lambda text: setattr(new_meta, 'studyDate', text))

    comment_ax = plt.axes((0.09, 0.05, 0.4, 0.05))
    comment_box = TextBox(comment_ax, 'Comment:', initial=new_meta.imageComments)
    comment_box.on_submit(lambda text: setattr(new_meta, 'imageComments', text))

    button_ax = plt.axes((0.80, 0.05, 0.15, 0.05))
    button = Button(button_ax, 'Save as DICOM')
    button.on_clicked(lambda event: save_dicom(inversed_sinograms[new_meta.filter][-1], new_meta))

    print(f'{rmse(np.copy(image_original), inversed_sinograms[0][-1]):0.3f} '
          f'{rmse(np.copy(image_original), inversed_sinograms[1][-1]):0.3f}')

    fig.tight_layout()
    plt.show()
