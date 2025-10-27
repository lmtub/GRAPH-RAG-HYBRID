GuestFileSeek *qmp_guest_file_seek(int64_t handle, int64_t offset,

                                   int64_t whence, Error **errp)

{

    GuestFileHandle *gfh;

    GuestFileSeek *seek_data;

    HANDLE fh;

    LARGE_INTEGER new_pos, off_pos;

    off_pos.QuadPart = offset;

    BOOL res;

    gfh = guest_file_handle_find(handle, errp);

    if (!gfh) {

        return NULL;

    }



    fh = gfh->fh;

    res = SetFilePointerEx(fh, off_pos, &new_pos, whence);

    if (!res) {

        error_setg_win32(errp, GetLastError(), "failed to seek file");

        return NULL;

    }

    seek_data = g_new0(GuestFileSeek, 1);

    seek_data->position = new_pos.QuadPart;

    return seek_data;

}
