"""Known-size sphere initialization from one RGB image and camera calibration.

No scene, object pose, depth or segmentation is accepted by this interface.
The launch-volume prior is the published Catch It! task distribution, padded
to accommodate pixel-level radius uncertainty, not an episode-specific crop.
"""
import cv2
import numpy as np

from event_sphere_tracker_contour_q80_occlusion import sphere_from_edge_pixels

cv2.setNumThreads(2)


def foreground_outline(image, center, radius):
    center=np.asarray(center,dtype=float);height,width=image.shape[:2]
    x0=max(0,int(center[0]-2.3*radius));y0=max(0,int(center[1]-2.3*radius))
    x1=min(width,int(center[0]+2.3*radius)+1);y1=min(height,int(center[1]+2.3*radius)+1)
    patch=np.asarray(image[y0:y1,x0:x1],dtype=float);yy,xx=np.mgrid[y0:y1,x0:x1]
    distance=np.hypot(xx+.5-center[0],yy+.5-center[1])
    annulus=patch[(distance>1.5*radius)&(distance<2.2*radius)]
    if len(annulus)<20:return None
    background=np.median(annulus,axis=0)
    scatter=float(np.median(np.linalg.norm(annulus-background,axis=1)))
    threshold=max(3.,4*scatter)
    mask=((np.linalg.norm(patch-background,axis=2)>threshold)&(distance<1.45*radius)).astype(np.uint8)
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    cy=int(np.clip(round(center[1]-.5)-y0,0,mask.shape[0]-1));cx=int(np.clip(round(center[0]-.5)-x0,0,mask.shape[1]-1))
    label=int(labels[cy,cx])
    if label==0 or stats[label,cv2.CC_STAT_AREA]<15:return None
    contours,_=cv2.findContours((labels==label).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    points=max(contours,key=cv2.contourArea).reshape(-1,2).astype(float)+[x0+.5,y0+.5]
    points=cv2.convexHull(points.astype(np.float32)).reshape(-1,2).astype(float)
    if len(points)<8:return None
    return dict(points=points,background_scatter=scatter,threshold_RGB_units=threshold,foreground_pixels=int(stats[label,cv2.CC_STAT_AREA]))


def detect_sphere(image, intrinsics, camera_to_world, radius_m):
    gray = cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), .6)
    edges = cv2.Canny(smooth, 25, 75)
    ys, xs = np.nonzero(edges)
    points = np.column_stack((xs, ys)).astype(float) + .5
    circles = cv2.HoughCircles(smooth, cv2.HOUGH_GRADIENT, dp=1,
                              minDist=8, param1=75, param2=7,
                              minRadius=3, maxRadius=32)
    K = np.asarray(intrinsics, dtype=float)
    T = np.asarray(camera_to_world, dtype=float)
    candidates = []
    if circles is not None:
        for u, v, r in circles[0]:
            distances = np.linalg.norm(points - [u, v], axis=1)
            selected = points[np.abs(distances - r) < 1.7]
            if len(selected) < 12:
                continue
            fit = sphere_from_edge_pixels(selected, K, radius_m)
            if fit is None or fit['angle_bins'] < 10:
                continue
            outline=foreground_outline(image,fit['uv'],fit['radius_px'])
            refinement='Canny edge fallback'
            if outline is not None:
                refined=sphere_from_edge_pixels(outline['points'],K,radius_m)
                if refined is not None and refined['angle_bins']>=10 and .75*fit['radius_px']<refined['radius_px']<1.35*fit['radius_px']:
                    fit=refined;refinement='estimated RGB foreground convex silhouette'
            world = (T @ np.r_[fit['camera_position'], 1.])[:3]
            if not (-.8 < world[0] < .8 and 1.8 < world[1] < 2.9
                    and .35 < world[2] < 1.95):
                continue
            yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
            dd = np.hypot(xx + .5 - fit['uv'][0], yy + .5 - fit['uv'][1])
            inside = gray[dd < .65 * fit['radius_px']]
            outside = gray[(dd > 1.4 * fit['radius_px']) &
                           (dd < 1.9 * fit['radius_px'])]
            if len(inside) == 0 or len(outside) == 0:
                continue
            contrast = abs(float(np.median(inside)) - float(np.median(outside)))
            if contrast < 12:
                continue
            score = contrast * fit['angle_bins'] / (.4 + fit['residual_px'])
            candidates.append(dict(position=world.tolist(), uv=fit['uv'].tolist(),
                                   radius_px=fit['radius_px'],
                                   residual_px=fit['residual_px'],
                                   contrast=contrast, score=score,refinement=refinement,radius_uncertainty_px=max(.35,fit['residual_px'])))
    if not candidates:
        return dict(valid=False, candidates=[], source='single actual RGB image')
    best = max(candidates, key=lambda c: c['score'])
    return dict(valid=True, **best, candidates=candidates,
                source='single actual RGB image; known-radius tangent-cone fit',
                camera_origin=T[:3, 3].tolist())
