#include <iostream>
#include <vector>
#include <fstream>
#include <string>
// #include <format> // Gali neveikti KTU serveryje, jei kompiliatorius per senas
#include <chrono>
#include <thread>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cstring> // Būtinas strncpy ir offsetof

#include <mpi.h>
#include <json.hpp>

//parunnint: 1) make -> 2) mpiexec -n 8 ./lygiagretusl2

using json = nlohmann::json;

#define DATA_MONITOR_SIZE 7
#define MAX_FOOD_NAME 50 // Maksimalus produkto pavadinimo ilgis

#define RANK_MAIN 0
#define RANK_DATA 1
#define RANK_RESULT 2
#define RANK_WORKER_START 3 // 3 ir daugiau yra Worker procesai

#define TAG_ADD_ITEM 100         // Main -> Data (siunčia Food)
#define TAG_REQUEST_ITEM 110     // Worker -> Data (prašo Food)
#define TAG_SEND_ITEM 120        // Data -> Worker (siunčia Food)
#define TAG_WORK_DONE 130        // Signalas apie darbo pabaigą
#define TAG_ADD_RESULT 140       // Worker -> Result (siunčia Result)
#define TAG_SEND_RESULTS_COUNT 150 // Result -> Main (siunčia rezultatų kiekį)
#define TAG_SEND_RESULTS_DATA 160  // Result -> Main (siunčia rezultatų masyvą)

// ---------- ATNAUJINTOS STRUKTŪROS ----------

struct Food {
    char name[MAX_FOOD_NAME];
    int quantity;
    double price;
};

struct Result {
    Food originalData;
    double calculatedValue; // Pervadinta iš computedData
};
// ------------------------------------------

// JSON deserializacija į Food struktūrą
void from_json(const json &j, Food &f) {
    std::string s_name;
    j.at("name").get_to(s_name);
    strncpy(f.name, s_name.c_str(), MAX_FOOD_NAME - 1);
    f.name[MAX_FOOD_NAME - 1] = '\0';
    j.at("quantity").get_to(f.quantity);
    j.at("price").get_to(f.price);
}

// ---------- NAUDOJAMA SKAIČIAVIMO LOGIKA ----------

// Skaičiavimo funkcija, naudojanti Food objektą
double calculateData(const Food& food) {
    const double VAT = 0.21;
    double total = food.price * food.quantity;
    total = total + total * VAT;

    double acc = 0.0;
    // Ilgas skaičiavimas
    for (long i = 0; i < 11100209; i++) {
        double angle = (food.quantity + i % 360) * 0.01745;
        acc += sin(angle) * cos(angle) + tan(angle + 0.1);
        if (acc > 1e6) acc = fmod(acc, 1000.0);
    }

    total *= (1.0 + fmod(acc, 0.2));

    // Grąžiname double, o ne int, kad būtų tiksliau filtruojant
    return total;
}

// Filtravimo funkcija
bool isValidResult(double totalPrice) {
    return totalPrice >= 50.0;
}
// -------------------------------------------------

// MPI tipo kūrimas Food struktūrai
void create_mpi_food_type(MPI_Datatype *mpi_food_type) {
    const int nitems = 3;
    int blocklengths[3] = {MAX_FOOD_NAME, 1, 1};
    MPI_Datatype types[3] = {MPI_CHAR, MPI_INT, MPI_DOUBLE};
    MPI_Aint offsets[3];

    offsets[0] = offsetof(Food, name);
    offsets[1] = offsetof(Food, quantity);
    offsets[2] = offsetof(Food, price);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, mpi_food_type);
    MPI_Type_commit(mpi_food_type);
}

// MPI tipo kūrimas Result struktūrai
void create_mpi_result_type(MPI_Datatype mpi_food_type, MPI_Datatype *mpi_result_type) {
    const int nitems = 2;
    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {mpi_food_type, MPI_DOUBLE}; // Naudojam mpi_food_type
    MPI_Aint offsets[2];

    offsets[0] = offsetof(Result, originalData);
    offsets[1] = offsetof(Result, calculatedValue);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, mpi_result_type);
    MPI_Type_commit(mpi_result_type);
}

// Rezultatų spausdinimas į failą [cite: 33]
void print_results_to_file(const std::string& result_file, Result* final_results, int count) {
    std::ofstream out_file(result_file);
    if (!out_file.is_open()) {
        std::cerr << "Klaida: nepavyko sukurti rezultatu failo." << std::endl;
        return;
    }

    out_file << std::left;
    out_file << "-----------------------------------------------------------------------------------------\n";
    out_file << "| " << std::setw(25) << "Produkto Pavadinimas"
             << "| " << std::setw(15) << "Kiekis"
             << "| " << std::setw(20) << "Kaina (vnt.)"
             << "| " << std::setw(20) << "Skaiciuota Reiksme" << "|\n";
    out_file << "-----------------------------------------------------------------------------------------\n";

    if (final_results == nullptr || count <= 0) {
        out_file << "|                               Nerasta rezultatu, atitinkanciu kriterijus                                |\n";
    } else {
        for (int i = 0; i < count; i++) {
            out_file << "| ";
            out_file << std::setw(25) << final_results[i].originalData.name;
            out_file << "| ";
            out_file << std::setw(15) << final_results[i].originalData.quantity;
            out_file << "| ";
            out_file << std::setw(20) << final_results[i].originalData.price;
            out_file << "| ";
            out_file << std::setw(20) << final_results[i].calculatedValue;
            out_file << "| ";
            out_file << std::endl;
        }
    }
    out_file << "-----------------------------------------------------------------------------------------\n";
    out_file.close();
    std::cout << "Rezultatai issaugoti faile '"<< result_file << "'." << std::endl;
}

// Pagrindinis procesas (MainThread) [cite: 6, 24]
void run_main_process(int num_workers, MPI_Datatype mpi_food_type, MPI_Datatype mpi_result_type) {
    const std::string filename = "IFF3_2_ValinciuteD_L1_dat_1.json";
    std::string result_file = "IFF3_2_ValinciuteD_L2_rez.txt";

    // 1. Nuskaito duomenų failą [cite: 25]
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "Klaida: nepavyko atidaryti duomenu failo '" << filename << "'" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
        return;
    }

    json j;
    f >> j;
    // Skaitome iš "foods" rakto, kaip pavyzdyje
    auto all_foods = j.at("foods").get<std::vector<Food>>();
    f.close();

    const auto startTime = std::chrono::high_resolution_clock::now();

    // 3. Duomenų masyvą valdančiam procesui po vieną persiunčia elementus [cite: 31]
    for (const auto &food : all_foods) {
        MPI_Send(&food, 1, mpi_food_type, RANK_DATA, TAG_ADD_ITEM, MPI_COMM_WORLD);
    }

    // Informuoja Data procesą, kad duomenų siuntimas baigtas
    MPI_Send(nullptr, 0, MPI_BYTE, RANK_DATA, TAG_WORK_DONE, MPI_COMM_WORLD);

    // 4. Iš rezultatų masyvą valdančio proceso gauna rezultatus [cite: 32]
    int result_count = 0;
    MPI_Status status;
    MPI_Recv(&result_count, 1, MPI_INT, RANK_RESULT, TAG_SEND_RESULTS_COUNT, MPI_COMM_WORLD, &status);

    Result* final_results = nullptr;
    if (result_count > 0) {
        final_results = new Result[result_count];
        MPI_Recv(final_results, result_count, mpi_result_type, RANK_RESULT, TAG_SEND_RESULTS_DATA, MPI_COMM_WORLD, &status);
    }

    const auto endTime = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = endTime - startTime;
    std::cout << "Visi procesai baige darba per " << elapsed.count() << "s" << std::endl;

    // 5. Gautus rezultatus išveda į tekstinį failą [cite: 33]
    print_results_to_file(result_file, final_results, result_count);

    delete[] final_results;
}

// Duomenų masyvą valdantis procesas (DataThread) [cite: 18, 29, 39]
void run_data_process(int num_workers, MPI_Datatype mpi_food_type) {
    // 1. Turi tik sau matomą masyvą [cite: 40]
    Food* buffer = new Food[DATA_MONITOR_SIZE];
    size_t elementCount = 0;
    size_t insertIndex = 0;
    size_t removeIndex = 0;

    bool main_done_adding = false;
    int workers_finished = 0;
    MPI_Status status;

    while (workers_finished < num_workers) {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;

        if (source == RANK_MAIN) {
            if (tag == TAG_ADD_ITEM) {
                // 2. Gali gauti žinutę, kad reikia įdėti įrašą [cite: 41]
                // 3. Jei duomenų masyvas yra pilnas, nepriima žinučių [cite: 42]
                if (elementCount == DATA_MONITOR_SIZE) {
                    // Masyvas pilnas, laukiame, kol darbininkas paprašys elemento
                    // [cite: 49] - užtikrinam, kad buferis gali prisipildyti/ištuštėti
                    MPI_Recv(nullptr, 0, MPI_BYTE, MPI_ANY_SOURCE, TAG_REQUEST_ITEM, MPI_COMM_WORLD, &status);
                    int worker_rank = status.MPI_SOURCE;

                    MPI_Send(&buffer[removeIndex], 1, mpi_food_type, worker_rank, TAG_SEND_ITEM, MPI_COMM_WORLD);
                    removeIndex = (removeIndex + 1) % DATA_MONITOR_SIZE;
                    --elementCount;
                }

                // Priimame elementą iš Main
                MPI_Recv(&buffer[insertIndex], 1, mpi_food_type, RANK_MAIN, TAG_ADD_ITEM, MPI_COMM_WORLD, &status);
                insertIndex = (insertIndex + 1) % DATA_MONITOR_SIZE;
                ++elementCount;

            } else if (tag == TAG_WORK_DONE) {
                // Main baigė siųsti duomenis
                MPI_Recv(nullptr, 0, MPI_BYTE, RANK_MAIN, TAG_WORK_DONE, MPI_COMM_WORLD, &status);
                main_done_adding = true;
            }
        }
        else if (source >= RANK_WORKER_START) {
            // 2. Gali gauti žinutę, kad reikia pašalinti įrašą [cite: 41]
            if (tag == TAG_REQUEST_ITEM) {
                // 4. Jei duomenų masyvas yra tuščias, nepriima žinučių [cite: 43]
                if (elementCount == 0) {
                    if (main_done_adding) {
                        // Buferis tuščias IR Main baigė darbą. Siunčiame WORK_DONE.
                        MPI_Recv(nullptr, 0, MPI_BYTE, source, TAG_REQUEST_ITEM, MPI_COMM_WORLD, &status);
                        MPI_Send(nullptr, 0, MPI_BYTE, source, TAG_WORK_DONE, MPI_COMM_WORLD);
                        workers_finished++;
                        continue;
                    } else {
                        // Buferis tuščias, bet Main dar dirba. Laukiame duomenų iš Main. [cite: 49]
                        MPI_Recv(&buffer[insertIndex], 1, mpi_food_type, RANK_MAIN, TAG_ADD_ITEM, MPI_COMM_WORLD, &status);
                        insertIndex = (insertIndex + 1) % DATA_MONITOR_SIZE;
                        ++elementCount;
                    }
                }

                // Aptarnaujame Worker prašymą
                MPI_Recv(nullptr, 0, MPI_BYTE, source, TAG_REQUEST_ITEM, MPI_COMM_WORLD, &status);
                MPI_Send(&buffer[removeIndex], 1, mpi_food_type, source, TAG_SEND_ITEM, MPI_COMM_WORLD);
                removeIndex = (removeIndex + 1) % DATA_MONITOR_SIZE;
                --elementCount;
            }
        }
    }

    delete[] buffer;
}

// Darbinis procesas (WorkerThread) [cite: 14, 28, 34]
void run_worker_process(int rank, MPI_Datatype mpi_food_type, MPI_Datatype mpi_result_type) {
    Food food; // Naudojam Food, o ne Planet
    MPI_Status status;

    while (true) {
        // 1. Iš duomenų masyvą valdančio proceso paprašo įrašo [cite: 35]
        MPI_Send(nullptr, 0, MPI_BYTE, RANK_DATA, TAG_REQUEST_ITEM, MPI_COMM_WORLD);

        // ...ir jį gauna (arba WORK_DONE signalą)
        MPI_Recv(&food, 1, mpi_food_type, RANK_DATA, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        // 4. Darbas kartojamas, kol bus apdoroti visi įrašai [cite: 38]
        if (status.MPI_TAG == TAG_WORK_DONE) {
            break; // Baigiame darbą [cite: 47]
        }

        // 2. Apskaičiuoja pasirinktos operacijos rezultatą: [cite: 36]
        const double total = calculateData(food);

        // 3. Jei gautas rezultatas tenkina pasirinktą kriterijų, siunčia jį [cite: 37]
        if (isValidResult(total)) {
            Result res = {food, total};
            MPI_Send(&res, 1, mpi_result_type, RANK_RESULT, TAG_ADD_RESULT, MPI_COMM_WORLD);
        }
    }

    // Pranešame Result procesui, kad baigėme darbą
    MPI_Send(nullptr, 0, MPI_BYTE, RANK_RESULT, TAG_WORK_DONE, MPI_COMM_WORLD);
}

// Rezultatų masyvą valdantis procesas (ResultThread) [cite: 19, 30]
void run_result_process(int num_workers, MPI_Datatype mpi_result_type) {
    // 1. Turi tik sau matomą masyvą [cite: 44]
    std::vector<Result> results;
    results.reserve(100);

    int workers_finished = 0;
    Result temp_result; //praso siusti rezultatus i sita
    MPI_Status status;

    // Rikiavimui (nuo didžiausio iki mažiausio)
    auto compare = [](const Result& a, const Result& b) {
        return a.calculatedValue > b.calculatedValue;
    };

    while (workers_finished < num_workers) {
        MPI_Recv(&temp_result, 1, mpi_result_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_SOURCE >= RANK_WORKER_START) {
            if (status.MPI_TAG == TAG_ADD_RESULT) {
                // 2. Turi galėti įterpti elementą [cite: 45]
                // Rūšiuojame iš karto (Insertion sort principu)
                auto it = std::upper_bound(results.begin(), results.end(), temp_result, compare);
                results.insert(it, temp_result);
            } else if (status.MPI_TAG == TAG_WORK_DONE) {
                workers_finished++;
            }
        }
    }

    // Visi darbininkai baigė.
    // 2. ...bei persiųsti esamus elementus pagrindiniam procesui [cite: 45]
    int result_count = static_cast<int>(results.size());
    MPI_Send(&result_count, 1, MPI_INT, RANK_MAIN, TAG_SEND_RESULTS_COUNT, MPI_COMM_WORLD);

    if (result_count > 0) {
        MPI_Send(results.data(), result_count, mpi_result_type, RANK_MAIN, TAG_SEND_RESULTS_DATA, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 2. Paleidžia procesus [cite: 26]
    // Reikia bent 4 procesų: Main, Data, Result, 1 Worker [cite: 5]
    if (world_size < 4) {
        if (rank == 0) {
            std::cerr << "Klaida: programai reikia bent 4 procesu (Main, Data, Result, 1+ Workers)." << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    int num_workers = world_size - 3; // [cite: 27]

    // Sukuriame MPI tipus
    MPI_Datatype mpi_food_type, mpi_result_type;
    create_mpi_food_type(&mpi_food_type);
    create_mpi_result_type(mpi_food_type, &mpi_result_type);

    if (rank == RANK_MAIN) {
        run_main_process(num_workers, mpi_food_type, mpi_result_type);
    } else if (rank == RANK_DATA) {
        run_data_process(num_workers, mpi_food_type);
    } else if (rank == RANK_RESULT) {
        run_result_process(num_workers, mpi_result_type);
    } else {
        // Visi kiti procesai yra darbininkai
        run_worker_process(rank, mpi_food_type, mpi_result_type);
    }

    // Atlaisviname MPI tipus
    MPI_Type_free(&mpi_food_type);
    MPI_Type_free(&mpi_result_type);

    MPI_Finalize();
    return 0;
}