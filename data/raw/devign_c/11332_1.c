static VFIOGroup *vfio_get_group(int groupid)

{

    VFIOGroup *group;

    char path[32];

    struct vfio_group_status status = { .argsz = sizeof(status) };



    QLIST_FOREACH(group, &group_list, next) {

        if (group->groupid == groupid) {

            return group;

        }

    }



    group = g_malloc0(sizeof(*group));



    snprintf(path, sizeof(path), "/dev/vfio/%d", groupid);

    group->fd = qemu_open(path, O_RDWR);

    if (group->fd < 0) {

        error_report("vfio: error opening %s: %m", path);

        goto free_group_exit;

    }



    if (ioctl(group->fd, VFIO_GROUP_GET_STATUS, &status)) {

        error_report("vfio: error getting group status: %m");

        goto close_fd_exit;

    }



    if (!(status.flags & VFIO_GROUP_FLAGS_VIABLE)) {

        error_report("vfio: error, group %d is not viable, please ensure "

                     "all devices within the iommu_group are bound to their "

                     "vfio bus driver.", groupid);

        goto close_fd_exit;

    }



    group->groupid = groupid;

    QLIST_INIT(&group->device_list);



    if (vfio_connect_container(group)) {

        error_report("vfio: failed to setup container for group %d", groupid);

        goto close_fd_exit;

    }



    if (QLIST_EMPTY(&group_list)) {

        qemu_register_reset(vfio_pci_reset_handler, NULL);

    }



    QLIST_INSERT_HEAD(&group_list, group, next);



    vfio_kvm_device_add_group(group);



    return group;



close_fd_exit:

    close(group->fd);



free_group_exit:

    g_free(group);



    return NULL;

}
