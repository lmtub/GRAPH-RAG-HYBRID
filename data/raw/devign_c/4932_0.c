static void virtio_rng_process(VirtIORNG *vrng)

{

    size_t size;



    if (!is_guest_ready(vrng)) {

        return;

    }



    size = get_request_size(vrng->vq);

    size = MIN(vrng->quota_remaining, size);

    if (size) {

        rng_backend_request_entropy(vrng->rng, size, chr_read, vrng);

    }

}
