static void prstatfs_to_statfs(struct statfs *stfs, ProxyStatFS *prstfs)

{

    memset(stfs, 0, sizeof(*stfs));

    stfs->f_type = prstfs->f_type;

    stfs->f_bsize = prstfs->f_bsize;

    stfs->f_blocks = prstfs->f_blocks;

    stfs->f_bfree = prstfs->f_bfree;

    stfs->f_bavail = prstfs->f_bavail;

    stfs->f_files = prstfs->f_files;

    stfs->f_ffree = prstfs->f_ffree;

    stfs->f_fsid.__val[0] = prstfs->f_fsid[0] & 0xFFFFFFFFU;

    stfs->f_fsid.__val[1] = prstfs->f_fsid[1] >> 32 & 0xFFFFFFFFU;

    stfs->f_namelen = prstfs->f_namelen;

    stfs->f_frsize = prstfs->f_frsize;

}
