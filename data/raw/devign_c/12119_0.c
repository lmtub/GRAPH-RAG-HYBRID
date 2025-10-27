static int vfio_base_device_init(VFIODevice *vbasedev)

{

    VFIOGroup *group;

    VFIODevice *vbasedev_iter;

    char path[PATH_MAX], iommu_group_path[PATH_MAX], *group_name;

    ssize_t len;

    struct stat st;

    int groupid;

    int ret;



    /* name must be set prior to the call */

    if (!vbasedev->name || strchr(vbasedev->name, '/')) {

        return -EINVAL;

    }



    /* Check that the host device exists */

    g_snprintf(path, sizeof(path), "/sys/bus/platform/devices/%s/",

               vbasedev->name);



    if (stat(path, &st) < 0) {

        error_report("vfio: error: no such host device: %s", path);

        return -errno;

    }



    g_strlcat(path, "iommu_group", sizeof(path));

    len = readlink(path, iommu_group_path, sizeof(iommu_group_path));

    if (len < 0 || len >= sizeof(iommu_group_path)) {

        error_report("vfio: error no iommu_group for device");

        return len < 0 ? -errno : -ENAMETOOLONG;

    }



    iommu_group_path[len] = 0;

    group_name = basename(iommu_group_path);



    if (sscanf(group_name, "%d", &groupid) != 1) {

        error_report("vfio: error reading %s: %m", path);

        return -errno;

    }



    trace_vfio_platform_base_device_init(vbasedev->name, groupid);



    group = vfio_get_group(groupid, &address_space_memory);

    if (!group) {

        error_report("vfio: failed to get group %d", groupid);

        return -ENOENT;

    }



    g_snprintf(path, sizeof(path), "%s", vbasedev->name);



    QLIST_FOREACH(vbasedev_iter, &group->device_list, next) {

        if (strcmp(vbasedev_iter->name, vbasedev->name) == 0) {

            error_report("vfio: error: device %s is already attached", path);

            vfio_put_group(group);

            return -EBUSY;

        }

    }

    ret = vfio_get_device(group, path, vbasedev);

    if (ret) {

        error_report("vfio: failed to get device %s", path);

        vfio_put_group(group);

        return ret;

    }



    ret = vfio_populate_device(vbasedev);

    if (ret) {

        error_report("vfio: failed to populate device %s", path);

        vfio_put_group(group);

    }



    return ret;

}
