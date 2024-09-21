import numpy as np
import cv2
import tqdm

def generate_pixels(img, SLIC_height, SLIC_width, SLIC_ITERATIONS, SLIC_centers, SLIC_labimg, step, SLIC_m, SLIC_clusters):
    indnp = np.mgrid[0:SLIC_height, 0:SLIC_width].swapaxes(0, 2).swapaxes(0, 1)
    for i in tqdm.tqdm(range(SLIC_ITERATIONS)):
        SLIC_distances = 1 * np.ones(img.shape[:2])
        for j in range(SLIC_centers.shape[0]):
            x_low, x_high = int(SLIC_centers[j][3] - step), int(SLIC_centers[j][3] + step)
            y_low, y_high = int(SLIC_centers[j][4] - step), int(SLIC_centers[j][4] + step)

            x_low = max(x_low, 0)
            x_high = min(x_high, SLIC_width)
            y_low = max(y_low, 0)
            y_high = min(y_high, SLIC_height)

            cropimg = SLIC_labimg[y_low:y_high, x_low:x_high]
            color_diff = cropimg - SLIC_labimg[int(SLIC_centers[j][4]), int(SLIC_centers[j][3])]
            color_distance = np.sqrt(np.sum(np.square(color_diff), axis=2))

            yy, xx = np.ogrid[y_low:y_high, x_low:x_high]
            pixdist = np.sqrt((yy - SLIC_centers[j][4])**2 + (xx - SLIC_centers[j][3])**2)

            dist = np.sqrt((color_distance / SLIC_m)**2 + (pixdist / step)**2)

            distance_crop = SLIC_distances[y_low:y_high, x_low:x_high]
            idx = dist < distance_crop
            distance_crop[idx] = dist[idx]
            SLIC_distances[y_low:y_high, x_low:x_high] = distance_crop
            SLIC_clusters[y_low:y_high, x_low:x_high][idx] = j

        for k in range(len(SLIC_centers)):
            idx = (SLIC_clusters == k)
            colornp = SLIC_labimg[idx]
            distnp = indnp[idx]
            SLIC_centers[k][:3] = np.sum(colornp, axis=0)
            sumy, sumx = np.sum(distnp, axis=0)
            SLIC_centers[k][3:] = sumx, sumy
            SLIC_centers[k] /= np.sum(idx)

def create_connectivity(img, SLIC_width, SLIC_height, SLIC_centers, SLIC_clusters):
    label = 0
    adj_label = 0
    lims = int(SLIC_width * SLIC_height / SLIC_centers.shape[0])
    
    new_clusters = -1 * np.ones(img.shape[:2], dtype=np.int64)
    elements = []
    for i in range(SLIC_width):
        for j in range(SLIC_height):
            if new_clusters[j, i] == -1:
                elements = [(j, i)]
                for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    x = elements[0][1] + dx
                    y = elements[0][0] + dy
                    if 0 <= x < SLIC_width and 0 <= y < SLIC_height and new_clusters[y, x] >= 0:
                        adj_label = new_clusters[y, x]

                count = 1
                counter = 0
                while counter < count:
                    for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                        x = elements[counter][1] + dx
                        y = elements[counter][0] + dy

                        if 0 <= x < SLIC_width and 0 <= y < SLIC_height:
                            if new_clusters[y, x] == -1 and SLIC_clusters[j, i] == SLIC_clusters[y, x]:
                                elements.append((y, x))
                                new_clusters[y, x] = label
                                count += 1
                    counter += 1

                if count <= lims >> 2:
                    for counter in range(count):
                        new_clusters[elements[counter]] = adj_label

                    label -= 1

                label += 1

    SLIC_new_clusters = new_clusters

def display_contours(filename, color, img, SLIC_width, SLIC_height, SLIC_clusters):
    is_taken = np.zeros(img.shape[:2], bool)
    contours = []

    for i in range(SLIC_width):
        for j in range(SLIC_height):
            nr_p = 0
            for dx, dy in [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]:
                x = i + dx
                y = j + dy
                if 0 <= x < SLIC_width and 0 <= y < SLIC_height:
                    if not is_taken[y, x] and SLIC_clusters[j, i] != SLIC_clusters[y, x]:
                        nr_p += 1

            if nr_p >= 2:
                is_taken[j, i] = True
                contours.append([j, i])
    
    np.save(filename, contours)

    for i in range(len(contours)):
        img[contours[i][0], contours[i][1]] = color

def find_local_minimum(center, SLIC_labimg):
    min_grad = np.inf
    loc_min = center
    for i in range(center[0] - 1, center[0] + 2):
        for j in range(center[1] - 1, center[1] + 2):
            c1 = SLIC_labimg[j + 1, i]
            c2 = SLIC_labimg[j, i + 1]
            c3 = SLIC_labimg[j, i]
            grad = np.abs(c1[0] - c3[0]) + np.abs(c2[0] - c3[0])
            if grad < min_grad:
                min_grad = grad
                loc_min = [i, j]
    return loc_min

def calculate_centers(step, SLIC_width, SLIC_height, SLIC_labimg):
    centers = []
    for i in range(step, SLIC_width - int(step / 2), step):
        for j in range(step, SLIC_height - int(step / 2), step):
            nc = find_local_minimum((i, j), SLIC_labimg)
            color = SLIC_labimg[nc[1], nc[0]]
            center = [color[0], color[1], color[2], nc[0], nc[1]]
            centers.append(center)
    return centers

def save_clusters_and_pixels_numpy(filename, SLIC_centers, SLIC_clusters):
    clusters = []
    
    for cluster_id in range(SLIC_centers.shape[0]):
        pixels = np.where(SLIC_clusters == cluster_id)
        cluster_pixels = np.stack((pixels[0], pixels[1]), axis=-1)
        clusters.append(cluster_pixels)
    
    clusters_array = np.array(clusters, dtype=object)
    np.save(filename, clusters_array)

def slic(img_path, filename_contours, filename_clusters, size, segment = 500, compactness=40, color = [0, 0, 0]):
    img = cv2.imread(img_path)
    img = cv2.resize(img, size)
    step = int((img.shape[0] * img.shape[1] / segment)**0.5)
    SLIC_m = int(compactness)
    SLIC_ITERATIONS = 4
    SLIC_height, SLIC_width = img.shape[:2]
    SLIC_labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    SLIC_distances = 1 * np.ones(img.shape[:2])
    SLIC_clusters = -1 * SLIC_distances
    SLIC_centers = np.array(calculate_centers(step, SLIC_width, SLIC_height, SLIC_labimg))

    generate_pixels(img, SLIC_height, SLIC_width, SLIC_ITERATIONS, SLIC_centers, SLIC_labimg, step, SLIC_m, SLIC_clusters)
    create_connectivity(img, SLIC_width, SLIC_height, SLIC_centers, SLIC_clusters)
    display_contours(filename_contours, color, img, SLIC_width, SLIC_height, SLIC_clusters)
    save_clusters_and_pixels_numpy(filename_clusters, SLIC_centers, SLIC_clusters)