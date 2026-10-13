#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/status.h>
#include <parquet/arrow/reader.h>
#include <iostream>
#include <string>


// https://mojoauth.com/dev-guides/serialize-and-deserialize-parquet-with-cpp
// Think this one might be really good to look at. We should have defo looked at this one -
// when making the geolife reader for data_processing
arrow::Status read_parquet(const std::string& path){
    ARROW_ASSIGN_OR_RAISE(auto in, arrow::io::ReadableFile::Open(path));
    ARROW_ASSIGN_OR_RAISE(auto reader, parquet::arrow::OpenFile(in, arrow::default_memory_pool()));
    ARROW_ASSIGN_OR_RAISE(auto read, reader->ReadTable());
    std::cout << read->schema()->ToString() << "\n";
    std::cout << "row groups: " << reader->num_row_groups() << "\n";

    return arrow::Status::OK();
}