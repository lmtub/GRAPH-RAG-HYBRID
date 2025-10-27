static void vfio_put_device(VFIOPCIDevice *vdev)

{

    QLIST_REMOVE(vdev, next);

    vdev->vbasedev.group = NULL;

    trace_vfio_put_device(vdev->vbasedev.fd);

    close(vdev->vbasedev.fd);

    g_free(vdev->vbasedev.name);

    if (vdev->msix) {

        g_free(vdev->msix);

        vdev->msix = NULL;

    }

}
