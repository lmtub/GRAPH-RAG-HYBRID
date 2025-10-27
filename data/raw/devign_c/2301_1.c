static void virtio_rng_device_realize(DeviceState *dev, Error **errp)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(dev);

    VirtIORNG *vrng = VIRTIO_RNG(dev);

    Error *local_err = NULL;



    if (!vrng->conf.period_ms > 0) {

        error_set(errp, QERR_INVALID_PARAMETER_VALUE, "period",

                  "a positive number");

        return;

    }



    if (vrng->conf.rng == NULL) {

        vrng->conf.default_backend = RNG_RANDOM(object_new(TYPE_RNG_RANDOM));



        user_creatable_complete(OBJECT(vrng->conf.default_backend),

                                &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            object_unref(OBJECT(vrng->conf.default_backend));

            return;

        }



        object_property_add_child(OBJECT(dev),

                                  "default-backend",

                                  OBJECT(vrng->conf.default_backend),

                                  NULL);



        /* The child property took a reference, we can safely drop ours now */

        object_unref(OBJECT(vrng->conf.default_backend));



        object_property_set_link(OBJECT(dev),

                                 OBJECT(vrng->conf.default_backend),

                                 "rng", NULL);

    }



    virtio_init(vdev, "virtio-rng", VIRTIO_ID_RNG, 0);



    vrng->rng = vrng->conf.rng;

    if (vrng->rng == NULL) {

        error_set(errp, QERR_INVALID_PARAMETER_VALUE, "rng", "a valid object");

        return;

    }



    vrng->vq = virtio_add_queue(vdev, 8, handle_input);



    /* Workaround: Property parsing does not enforce unsigned integers,

     * So this is a hack to reject such numbers. */

    if (vrng->conf.max_bytes > INT64_MAX) {

        error_set(errp, QERR_INVALID_PARAMETER_VALUE, "max-bytes",

                  "a non-negative integer below 2^63");

        return;

    }

    vrng->quota_remaining = vrng->conf.max_bytes;



    vrng->rate_limit_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL,

                                               check_rate_limit, vrng);



    timer_mod(vrng->rate_limit_timer,

                   qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL) + vrng->conf.period_ms);



    register_savevm(dev, "virtio-rng", -1, 1, virtio_rng_save,

                    virtio_rng_load, vrng);

}
