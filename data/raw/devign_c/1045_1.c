static void s390_virtio_rng_instance_init(Object *obj)

{

    VirtIORNGS390 *dev = VIRTIO_RNG_S390(obj);

    object_initialize(&dev->vdev, sizeof(dev->vdev), TYPE_VIRTIO_RNG);

    object_property_add_child(obj, "virtio-backend", OBJECT(&dev->vdev), NULL);

    object_property_add_link(obj, "rng", TYPE_RNG_BACKEND,

                             (Object **)&dev->vdev.conf.rng,


                             OBJ_PROP_LINK_UNREF_ON_RELEASE, NULL);

}