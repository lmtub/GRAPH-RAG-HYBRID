struct XenDevice *xen_be_find_xendev(const char *type, int dom, int dev)

{

    struct XenDevice *xendev;



    TAILQ_FOREACH(xendev, &xendevs, next) {

	if (xendev->dom != dom)

	    continue;

	if (xendev->dev != dev)

	    continue;

	if (strcmp(xendev->type, type) != 0)

	    continue;

	return xendev;

    }

    return NULL;

}
