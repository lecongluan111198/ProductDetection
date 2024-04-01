import numpy as np
from sklearn.cluster import DBSCAN
import cv2
import functools
import hdbscan

def estimate_angle(image, threshold1=50, threshold2=150):
    """
    Estimate the rotation angle of an image.

    Parameters:
    image (np.array): The image
    threshold1, threshold2 (int): Thresholds for the Canny edge detection

    Returns:
    float: The estimated rotation angle in degrees
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2)

    # Perform Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    # Calculate the angles of each line
    angles = [np.degrees(line[0][1]) for line in lines]

    # Return the median angle
    return np.median(angles)

def rotate_image(image, angle):
    """
    Rotate an image.

    Parameters:
    image (np.array): The image to rotate
    angle (float): The rotation angle in degrees

    Returns:
    np.array: The rotated image
    """
    # Get image dimensions
    (h, w) = image.shape[:2]

    # Compute the center of the image
    center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    Parameters:
    origin (tuple): The origin point (x, y)
    point (tuple): The point to rotate (x, y)
    angle (float): The rotation angle in degrees

    Returns:
    tuple: The rotated point (x, y)
    """
    # Convert angle from degrees to radians
    angle = np.deg2rad(angle)

    # Translate point back to origin
    point = (point[0] - origin[0], point[1] - origin[1])

    # Perform rotation
    new_point = (point[0]*np.cos(angle) - point[1]*np.sin(angle), point[0]*np.sin(angle) + point[1]*np.cos(angle))

    # Translate point back
    new_point = (new_point[0] + origin[0], new_point[1] + origin[1])

    return new_point


map = {
    '16705': [{"x":0.6121445652329441,"y":0,"width":0.08468549679487179,"height":0.45992782711982727,"cls":"Nanh1L"},{"x":0.7374076647636217,"y":0.5443059206008911,"width":0.092999021648924,"height":0.43414419889450073,"cls":"CookVitamin1L"},{"x":0.5372281811612867,"y":0.02228500135242939,"width":0.0751077169900412,"height":0.43154908530414104,"cls":"Nanh1L"},{"x":0.6541689023866758,"y":0.5383492708206177,"width":0.08594295515682235,"height":0.4022015929222107,"cls":"CookVitamin1L"},{"x":0.5823608845581502,"y":0.530277669429779,"width":0.0743601816477793,"height":0.3743462562561035,"cls":"CookVitamin1L"},{"x":0.8308060754349816,"y":0.5570147037506104,"width":0.10633468907394689,"height":0.44298529624938965,"cls":"CookVitamin1L"},{"x":0.37612662402701463,"y":0.0947294533252716,"width":0.05344050480769231,"height":0.35060080885887146,"cls":"Nanh1L"},{"x":0.6978196364182693,"y":0,"width":0.09942443624084249,"height":0.4662322402000427,"cls":"Nanh1L"},{"x":0.4741516336853251,"y":0.052007753401994705,"width":0.06197916666666667,"height":0.39752666279673576,"cls":"Nanh1L"},{"x":0.42620219136332416,"y":0.07630961388349533,"width":0.0526247800051511,"height":0.37130550295114517,"cls":"Nanh1L"},{"x":0.8073111621451465,"y":0,"width":0.1063734117445055,"height":0.4644233286380768,"cls":"Nanh1L"},{"x":0.4113947494562729,"y":0.5085783004760742,"width":0.053607781378777475,"height":0.30789095163345337,"cls":"CookVitamin1L"},{"x":0.46033796932234433,"y":0.5153313279151917,"width":0.05835604824862638,"height":0.3239938020706177,"cls":"CookVitamin1L"},{"x":0.3674592517671131,"y":0.5042039155960083,"width":0.049676468957474816,"height":0.2938971519470215,"cls":"CookVitamin1L"},{"x":0.3217834919800252,"y":0.49718061089515686,"width":0.049201669099130034,"height":0.28370359539985657,"cls":"CookVitamin1L"},{"x":0.33657398712940706,"y":0.11489011347293854,"width":0.04306611560639881,"height":0.33194978535175323,"cls":"Nanh1L"},{"x":0.2608381893171932,"y":0.14322003722190857,"width":0.041634807656536175,"height":0.30609238147735596,"cls":"Nanh1L"},{"x":0.1944391383356227,"y":0.16444697976112366,"width":0.040475922189789376,"height":0.284058541059494,"cls":"Nanh1L"},{"x":0.22646386003319596,"y":0.4899406433105469,"width":0.038980717362065015,"height":0.24607938528060913,"cls":"CookVitamin1L"},{"x":0.2965804438887935,"y":0.12971548736095428,"width":0.043369659312042126,"height":0.31744955480098724,"cls":"Nanh1L"},{"x":0.2270142048706502,"y":0.15744975209236145,"width":0.04151736528445513,"height":0.29182007908821106,"cls":"Nanh1L"},{"x":0.2849059835021749,"y":0.4935821294784546,"width":0.040268380301339286,"height":0.27077436447143555,"cls":"CookVitamin1L"},{"x":0.5143488063043727,"y":0.524080216884613,"width":0.07112554193853023,"height":0.3497992157936096,"cls":"CookVitamin1L"},{"x":0.2607623313372825,"y":0.4910130798816681,"width":0.032581147693452384,"height":0.2626260221004486,"cls":"CookVitamin1L"},{"x":0.14942528735632185,"y":0.649025069637883,"width":0.04075235109717868,"height":0.23537604456824512,"cls":"Cook2L"},{"x":0.09194258246229682,"y":0.6467714905738831,"width":0.05701461065383184,"height":0.23165184259414673,"cls":"Cook2L"}],
    'test_1': [{"x": 0.5266340970993042, "y": 0.33533957600593567, "width": 0.18304495016733804, "height": 0.2910570800304413, "cls": "Nanh1L", "conf": 0.9703866839408875}, {"x": 0.6784218152364095, "y": 0.6771976351737976, "width": 0.1540060043334961, "height": 0.2366352081298828, "cls": "Nanh1L", "conf": 0.968877911567688}, {"x": 0.17536568641662598, "y": 0.36784547567367554, "width": 0.21730291843414307, "height": 0.3093477487564087, "cls": "Nanh1L", "conf": 0.9673972725868225}, {"x": 0.5867050886154175, "y": 0.7019793391227722, "width": 0.14762675762176514, "height": 0.2303737998008728, "cls": "Nanh1L", "conf": 0.9673548340797424}, {"x": 0.4267019033432007, "y": 0.36505380272865295, "width": 0.17047548294067383, "height": 0.2842663824558258, "cls": "Nanh1L", "conf": 0.9659744501113892}, {"x": 0.3840154806772868, "y": 0.718009352684021, "width": 0.17249584197998047, "height": 0.2118934988975525, "cls": "Nanh1L", "conf": 0.9652544260025024}, {"x": 0.30098505814870197, "y": 0.3998568058013916, "width": 0.19685864448547363, "height": 0.2706490755081177, "cls": "Nanh1L", "conf": 0.9647661447525024}, {"x": 0.4957342942555745, "y": 0.7190680503845215, "width": 0.1529541015625, "height": 0.22063958644866943, "cls": "Nanh1L", "conf": 0.9623481035232544}],
    'test_2': [{"x": 0.08738041917483012, "y": 0.17826585471630096, "width": 0.10197777549425761, "height": 0.2419043630361557, "cls": "Cook1L", "conf": 0.9685494303703308}, {"x": 0.672705888748169, "y": 0.12969303131103516, "width": 0.08230527242024739, "height": 0.25713759660720825, "cls": "Nanh1L", "conf": 0.9652261734008789}, {"x": 0.37923868497212726, "y": 0.09191960096359253, "width": 0.09848674138387044, "height": 0.31751924753189087, "cls": "Nanh1L", "conf": 0.9645372033119202}, {"x": 0.49856066703796387, "y": 0.5610998868942261, "width": 0.1344000498453776, "height": 0.26084423065185547, "cls": "CookVitamin1L", "conf": 0.9603537917137146}, {"x": 0.4194657802581787, "y": 0.5760003924369812, "width": 0.1258627971013387, "height": 0.2788979411125183, "cls": "CookVitamin1L", "conf": 0.9573801159858704}, {"x": 0.46054553985595703, "y": 0.10616709291934967, "width": 0.10292458534240723, "height": 0.3017951399087906, "cls": "Nanh1L", "conf": 0.9558059573173523}, {"x": 0.1920118530591329, "y": 0.09371143579483032, "width": 0.08932652076085408, "height": 0.3130149245262146, "cls": "Cook1L", "conf": 0.9466367363929749}, {"x": 0.6021474997202555, "y": 0.12336856126785278, "width": 0.09383908907572429, "height": 0.27013686299324036, "cls": "Nanh1L", "conf": 0.9334663152694702}, {"x": 0.5390285650889078, "y": 0.11241523176431656, "width": 0.09698967138926189, "height": 0.2875668928027153, "cls": "Nanh1L", "conf": 0.9304478764533997}, {"x": 0.27950404087702435, "y": 0.5877030491828918, "width": 0.1553150216738383, "height": 0.2962966561317444, "cls": "CookVitamin1L", "conf": 0.9195604920387268}, {"x": 0.5697750647862753, "y": 0.5389846563339233, "width": 0.1390612522761027, "height": 0.25318634510040283, "cls": "CookVitamin1L", "conf": 0.902631402015686}, {"x": 0.27492841084798175, "y": 0.07842065393924713, "width": 0.115334947903951, "height": 0.3400464504957199, "cls": "Nanh1L", "conf": 0.890117347240448}],
    'test_3': [{"x": 0.4781730572382609, "y": 0.46958431601524353, "width": 0.0910944143931071, "height": 0.2071622908115387, "cls": "Nanh1L", "conf": 0.9808716177940369}, {"x": 0.6438210805257162, "y": 0.3942832946777344, "width": 0.09365487098693848, "height": 0.25279104709625244, "cls": "Nanh1L", "conf": 0.9793457984924316}, {"x": 0.3769862651824951, "y": 0.46842774748802185, "width": 0.10509443283081055, "height": 0.21376708149909973, "cls": "Nanh1L", "conf": 0.9791402220726013}, {"x": 0.5616310040156046, "y": 0.4606338441371918, "width": 0.08969128131866455, "height": 0.20655491948127747, "cls": "Nanh1L", "conf": 0.9787188172340393}, {"x": 0.4330863157908122, "y": 0.7096388936042786, "width": 0.09898948669433594, "height": 0.14629459381103516, "cls": "Nanh1L", "conf": 0.9341750741004944}, {"x": 0.5213476816813151, "y": 0.7023447155952454, "width": 0.07239723205566406, "height": 0.14405423402786255, "cls": "Nanh1L", "conf": 0.9315048456192017}, {"x": 0.5828177134195963, "y": 0.6936925649642944, "width": 0.06702232360839844, "height": 0.13664811849594116, "cls": "Nanh1L", "conf": 0.9155064225196838}, {"x": 0.3644050757090251, "y": 0.7033047676086426, "width": 0.11033737659454346, "height": 0.14838439226150513, "cls": "Nanh1L", "conf": 0.7995988130569458}]
}
# https://stc2.kido.vn/image_upload/TAC/merchandising/202311/231121/36267_1700558971533.jpg //difficult to merge line
# https://stc2.kido.vn/image_upload/TAC/merchandising/202311/231122/36267_1700648200209.jpg //difficult to merge line
# https://stc2.kido.vn/image_upload/TAC/merchandising/202311/231118/543_1700282179392.jpg //difficult to merge line
# https://stc2.kido.vn/image_upload/TAC/merchandising/202311/231120/36267_1700466067899.jpg //difficult to merge line
# https://stc2.kido.vn/image_upload/TAC/merchandising/202311/231127/3040_1701051723304.jpg //difficult to merge line
# https://stc2.kido.vn/image_upload/TAC/merchandising/202311/231116/1938_1700100059429.jpg //difficult to merge line
# https://stc2.kido.vn/image_upload/TAC/merchandising/202311/231122/1820_1700628342338.jpg //difficult to merge line

# https://stc2.kido.vn/image_upload/TAC/merchandising/202311/231118/35720_1700280615437.jpg //different lines
# https://stc2.kido.vn/image_upload/TAC/merchandising/202311/231116/35720_1700128102593.jpg //different lines
# https://stc2.kido.vn/image_upload/TAC/merchandising/202311/231123/35136_1700710401230.jpg //different lines
# https://stc2.kido.vn/image_upload/TAC/merchandising/202311/231122/34637_1700624563438.jpg //different lines
# https://stc2.kido.vn/image_upload/TAC/merchandising/202311/231122/34637_1700624563438.jpg //difficult to merge line + different lines
def is_cluster(cluster1, cluster2, type, eps, min_samples, i, j):
    if type == 0:
        points = [y for x, y, w, h, cls in cluster1] + [y for x, y, w, h, cls in cluster2]
    elif type == 1:
        points = [y + h for x, y, w, h, cls in cluster1] + [y + h for x, y, w, h, cls in cluster2]
    else:
        points = [y + h/2 for x, y, w, h, cls in cluster1] + [y + h/2 for x, y, w, h, cls in cluster2]
    points = np.array(points).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    return len(np.unique(clustering.labels_)) == 1

def is_overlap(cluster1, cluster2, i, j):
    box1 = cluster1[len(cluster1) - 1]
    box2 = cluster2[0]
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]
    # print(box1)
    # print(box2)
    iou = cal_iou(box1, box2)
    print('overlap', iou ,i ,j)
    return iou >= 0.01

def cal_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    w = xB - xA
    h = yB - yA
    if w < 0 or h < 0:
        return 0.0
    # compute the area of intersection rectangle
    interArea = max(0, w + 1) * max(0, h + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def merge_cluster(clusters, eps=20, min_samples=1):
    n_cluster = len(clusters.values())
    merged = {}
    for i in range(n_cluster):
        if len(clusters[i]) == 0:
            continue
        merged[i] = clusters[i]
        for j in range(i + 1, n_cluster):
            if is_cluster(clusters[i], clusters[j], 0, eps, min_samples, i, j):
                merged[i] += clusters[j]
                clusters[j] = []
            elif is_cluster(clusters[i], clusters[j], 1, eps, min_samples, i, j):
                merged[i] += clusters[j]
                clusters[j] = []
            elif is_overlap(clusters[i], clusters[j], i, j):
                merged[i] += clusters[j]
                clusters[j] = []
    return merged

def inline():
    key = 'test_3'
    labels = map[key]
    def compare(v1, v2):
        return v1['x'] - v2['x']
    labels = sorted(labels, key=functools.cmp_to_key(compare))
    # print(labels)
    image = cv2.imread(f'/Users/luanlc/Documents/kido/AI/{key}.jpeg', cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = image.shape
    bboxes = [(l['x'] * img_width, l['y'] * img_height, l['width'] * img_width, l['height'] * img_height, l['cls']) for l in labels]
    centers = [(x + w / 2, y + h / 2, cls) for x, y, w, h, cls in bboxes]
    x_centers, y_centers, clses = zip(*centers)
    # print(centers)
    eps=20
    min_samples=1
    points = np.array(y_centers).reshape(-1, 1)
    # points = np.array([y for x, y, w, h, cls in bboxes]).reshape(-1, 1)
    # points = np.array([y + h for x, y, w, h, cls in bboxes]).reshape(-1, 1)
    # points = [[x, y] for x, y in zip(x_centers, y_centers)]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)  # Perform DBSCAN on y-coordinates
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
    # clustering = clusterer.fit_predict(points)
    clusters = {}
    for label, bbox in zip(clustering.labels_, bboxes):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(bbox)
    
    print(len(clusters))
    clusters = merge_cluster(clusters, eps, min_samples)
    print(len(clusters))
    def compare_cluster(v1, v2):
        return v1[1] - v2[1]
    clusters_sorted = sorted([(label, np.min([bb[1] for bb in bbs]), bbs) for label, bbs in clusters.items()], key=functools.cmp_to_key(compare_cluster))
    
    # line_bboxes = []
    for label, max_y, bbs in clusters_sorted:
        # line_bboxes.append([bb[4] for bb in bbs])
        print([(bb[4], bb[1]) for bb in bbs])
    # print(line_bboxes)
    # cv2.imshow('name', image) 
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows() 

inline()