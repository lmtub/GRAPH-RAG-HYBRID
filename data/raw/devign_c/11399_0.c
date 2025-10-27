int uuid_is_null(const uuid_t uu)

{

    uuid_t null_uuid = { 0 };

    return memcmp(uu, null_uuid, sizeof(uuid_t)) == 0;

}
