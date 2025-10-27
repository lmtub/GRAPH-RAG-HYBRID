static int hdev_open(BlockDriverState *bs, const char *filename, int flags)

{

    BDRVRawState *s = bs->opaque;

    int access_flags, create_flags;

    DWORD overlapped;

    char device_name[64];



    if (strstart(filename, "/dev/cdrom", NULL)) {

        if (find_cdrom(device_name, sizeof(device_name)) < 0)

            return -ENOENT;

        filename = device_name;

    } else {

        /* transform drive letters into device name */

        if (((filename[0] >= 'a' && filename[0] <= 'z') ||

             (filename[0] >= 'A' && filename[0] <= 'Z')) &&

            filename[1] == ':' && filename[2] == '\0') {

            snprintf(device_name, sizeof(device_name), "\\\\.\\%c:", filename[0]);

            filename = device_name;

        }

    }

    s->type = find_device_type(bs, filename);

    

    if ((flags & BDRV_O_ACCESS) == O_RDWR) {

        access_flags = GENERIC_READ | GENERIC_WRITE;

    } else {

        access_flags = GENERIC_READ;

    }

    create_flags = OPEN_EXISTING;



#ifdef QEMU_TOOL

    overlapped = FILE_ATTRIBUTE_NORMAL;

#else

    overlapped = FILE_FLAG_OVERLAPPED;

#endif

    s->hfile = CreateFile(filename, access_flags, 

                          FILE_SHARE_READ, NULL,

                          create_flags, overlapped, NULL);

    if (s->hfile == INVALID_HANDLE_VALUE) 

        return -1;

    return 0;

}
