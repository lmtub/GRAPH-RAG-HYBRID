static bool ufd_version_check(int ufd)

{

    struct uffdio_api api_struct;

    uint64_t ioctl_mask;



    api_struct.api = UFFD_API;

    api_struct.features = 0;

    if (ioctl(ufd, UFFDIO_API, &api_struct)) {

        error_report("postcopy_ram_supported_by_host: UFFDIO_API failed: %s",

                     strerror(errno));

        return false;

    }



    ioctl_mask = (__u64)1 << _UFFDIO_REGISTER |

                 (__u64)1 << _UFFDIO_UNREGISTER;

    if ((api_struct.ioctls & ioctl_mask) != ioctl_mask) {

        error_report("Missing userfault features: %" PRIx64,

                     (uint64_t)(~api_struct.ioctls & ioctl_mask));

        return false;

    }



    if (getpagesize() != ram_pagesize_summary()) {

        bool have_hp = false;

        /* We've got a huge page */

#ifdef UFFD_FEATURE_MISSING_HUGETLBFS

        have_hp = api_struct.features & UFFD_FEATURE_MISSING_HUGETLBFS;

#endif

        if (!have_hp) {

            error_report("Userfault on this host does not support huge pages");

            return false;

        }

    }

    return true;

}
