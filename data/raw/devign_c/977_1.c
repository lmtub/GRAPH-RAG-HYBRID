void vfio_put_group(VFIOGroup *group)

{

    if (!QLIST_EMPTY(&group->device_list)) {

        return;

    }



    vfio_kvm_device_del_group(group);

    vfio_disconnect_container(group);

    QLIST_REMOVE(group, next);

    trace_vfio_put_group(group->fd);

    close(group->fd);

    g_free(group);



    if (QLIST_EMPTY(&vfio_group_list)) {

        qemu_unregister_reset(vfio_reset_handler, NULL);

    }

}
