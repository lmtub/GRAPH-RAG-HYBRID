void desc_ring_free(DescRing *ring)

{

    if (ring->info) {

        g_free(ring->info);

    }

    g_free(ring);

}
