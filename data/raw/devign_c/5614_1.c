void unregister_savevm(DeviceState *dev, const char *idstr, void *opaque)

{

    SaveStateEntry *se, *new_se;

    char id[256] = "";



    if (dev && dev->parent_bus && dev->parent_bus->info->get_dev_path) {

        char *path = dev->parent_bus->info->get_dev_path(dev);

        if (path) {

            pstrcpy(id, sizeof(id), path);

            pstrcat(id, sizeof(id), "/");

            qemu_free(path);



    pstrcat(id, sizeof(id), idstr);



    QTAILQ_FOREACH_SAFE(se, &savevm_handlers, entry, new_se) {

        if (strcmp(se->idstr, id) == 0 && se->opaque == opaque) {

            QTAILQ_REMOVE(&savevm_handlers, se, entry);




            qemu_free(se);


