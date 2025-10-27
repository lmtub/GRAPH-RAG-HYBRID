int avpriv_lock_avformat(void)

{

    if (lockmgr_cb) {

        if ((*lockmgr_cb)(&avformat_mutex, AV_LOCK_OBTAIN))

            return -1;

    }

    return 0;

}
