static inline abi_ulong do_shmat(int shmid, abi_ulong shmaddr, int shmflg)

{

    abi_long raddr;

    void *host_raddr;

    struct shmid_ds shm_info;

    int i,ret;



    /* find out the length of the shared memory segment */

    ret = get_errno(shmctl(shmid, IPC_STAT, &shm_info));

    if (is_error(ret)) {

        /* can't get length, bail out */

        return ret;

    }



    mmap_lock();



    if (shmaddr)

        host_raddr = shmat(shmid, (void *)g2h(shmaddr), shmflg);

    else {

        abi_ulong mmap_start;



        mmap_start = mmap_find_vma(0, shm_info.shm_segsz);



        if (mmap_start == -1) {

            errno = ENOMEM;

            host_raddr = (void *)-1;

        } else

            host_raddr = shmat(shmid, g2h(mmap_start), shmflg | SHM_REMAP);

    }



    if (host_raddr == (void *)-1) {

        mmap_unlock();

        return get_errno((long)host_raddr);

    }

    raddr=h2g((unsigned long)host_raddr);



    page_set_flags(raddr, raddr + shm_info.shm_segsz,

                   PAGE_VALID | PAGE_READ |

                   ((shmflg & SHM_RDONLY)? 0 : PAGE_WRITE));



    for (i = 0; i < N_SHM_REGIONS; i++) {

        if (!shm_regions[i].in_use) {

            shm_regions[i].in_use = true;

            shm_regions[i].start = raddr;

            shm_regions[i].size = shm_info.shm_segsz;

            break;

        }

    }



    mmap_unlock();

    return raddr;



}
