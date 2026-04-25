def fit_best_ellipse_from_mask(mask, min_area=1000):
    """
    Finds the best ellipse from a binary marker mask.
    Returns ellipse and selected contour.
    
    ellipse format:
    ((center_x, center_y), (major_axis, minor_axis), angle)
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # fitEllipse needs at least 5 contour points
        if area > min_area and len(cnt) >= 5:
            valid_contours.append(cnt)

    if len(valid_contours) == 0:
        return None, None

    # Usually the marker is the largest coloured object
    best_contour = max(valid_contours, key=cv2.contourArea)

    ellipse = cv2.fitEllipse(best_contour)

    return ellipse, best_contour