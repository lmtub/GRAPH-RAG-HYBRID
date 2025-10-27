static int raw_open(BlockDriverState *bs, const char *filename, int flags)

{

    BDRVRawState *s = bs->opaque;

    int access_flags, create_flags;

    DWORD overlapped;



    s->type = FTYPE_FILE;



    if ((flags & BDRV_O_ACCESS) == O_RDWR) {

        access_flags = GENERIC_READ | GENERIC_WRITE;

    } else {

        access_flags = GENERIC_READ;

    }

    if (flags & BDRV_O_CREAT) {

        create_flags = CREATE_ALWAYS;

    } else {

        create_flags = OPEN_EXISTING;

    }

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
