static void rng_egd_free_request(RngRequest *req)

{

    g_free(req->data);

    g_free(req);

}
