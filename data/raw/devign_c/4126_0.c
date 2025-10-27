VirtIODevice *virtio_9p_init(DeviceState *dev, V9fsConf *conf)

 {

    V9fsState *s;

    int i, len;

    struct stat stat;

    FsTypeEntry *fse;





    s = (V9fsState *)virtio_common_init("virtio-9p",

                                    VIRTIO_ID_9P,

                                    sizeof(struct virtio_9p_config)+

                                    MAX_TAG_LEN,

                                    sizeof(V9fsState));



    /* initialize pdu allocator */

    QLIST_INIT(&s->free_list);

    for (i = 0; i < (MAX_REQ - 1); i++) {

	QLIST_INSERT_HEAD(&s->free_list, &s->pdus[i], next);

    }



    s->vq = virtio_add_queue(&s->vdev, MAX_REQ, handle_9p_output);



    fse = get_fsdev_fsentry(conf->fsdev_id);



    if (!fse) {

        /* We don't have a fsdev identified by fsdev_id */

        fprintf(stderr, "Virtio-9p device couldn't find fsdev "

                    "with the id %s\n", conf->fsdev_id);

        exit(1);

    }



    if (!fse->path || !conf->tag) {

        /* we haven't specified a mount_tag or the path */

        fprintf(stderr, "fsdev with id %s needs path "

                "and Virtio-9p device needs mount_tag arguments\n",

                conf->fsdev_id);

        exit(1);

    }



    if (!strcmp(fse->security_model, "passthrough")) {

        /* Files on the Fileserver set to client user credentials */

        s->ctx.fs_sm = SM_PASSTHROUGH;

    } else if (!strcmp(fse->security_model, "mapped")) {

        /* Files on the fileserver are set to QEMU credentials.

         * Client user credentials are saved in extended attributes.

         */

        s->ctx.fs_sm = SM_MAPPED;

    } else if (!strcmp(fse->security_model, "none")) {

        /*

         * Files on the fileserver are set to QEMU credentials.

         */

        s->ctx.fs_sm = SM_NONE;



    } else {

        fprintf(stderr, "Default to security_model=none. You may want"

                " enable advanced security model using "

                "security option:\n\t security_model=passthrough \n\t "

                "security_model=mapped\n");

        s->ctx.fs_sm = SM_NONE;

    }



    if (lstat(fse->path, &stat)) {

        fprintf(stderr, "share path %s does not exist\n", fse->path);

        exit(1);

    } else if (!S_ISDIR(stat.st_mode)) {

        fprintf(stderr, "share path %s is not a directory \n", fse->path);

        exit(1);

    }



    s->ctx.fs_root = qemu_strdup(fse->path);

    len = strlen(conf->tag);

    if (len > MAX_TAG_LEN) {

        len = MAX_TAG_LEN;

    }

    /* s->tag is non-NULL terminated string */

    s->tag = qemu_malloc(len);

    memcpy(s->tag, conf->tag, len);

    s->tag_len = len;

    s->ctx.uid = -1;



    s->ops = fse->ops;

    s->vdev.get_features = virtio_9p_get_features;

    s->config_size = sizeof(struct virtio_9p_config) +

                        s->tag_len;

    s->vdev.get_config = virtio_9p_get_config;



    return &s->vdev;

}
