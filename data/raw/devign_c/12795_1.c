static const char *search_keyval(const TiffGeoTagKeyName *keys, int n, int id)

{

    return ((TiffGeoTagKeyName*)bsearch(&id, keys, n, sizeof(keys[0]), cmp_id_key))->name;

}
