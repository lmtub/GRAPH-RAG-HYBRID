static void prstat_to_stat(struct stat *stbuf, ProxyStat *prstat)

{

   memset(stbuf, 0, sizeof(*stbuf));

   stbuf->st_dev = prstat->st_dev;

   stbuf->st_ino = prstat->st_ino;

   stbuf->st_nlink = prstat->st_nlink;

   stbuf->st_mode = prstat->st_mode;

   stbuf->st_uid = prstat->st_uid;

   stbuf->st_gid = prstat->st_gid;

   stbuf->st_rdev = prstat->st_rdev;

   stbuf->st_size = prstat->st_size;

   stbuf->st_blksize = prstat->st_blksize;

   stbuf->st_blocks = prstat->st_blocks;

   stbuf->st_atim.tv_sec = prstat->st_atim_sec;

   stbuf->st_atim.tv_nsec = prstat->st_atim_nsec;

   stbuf->st_mtime = prstat->st_mtim_sec;

   stbuf->st_mtim.tv_nsec = prstat->st_mtim_nsec;

   stbuf->st_ctime = prstat->st_ctim_sec;

   stbuf->st_ctim.tv_nsec = prstat->st_ctim_nsec;

}
