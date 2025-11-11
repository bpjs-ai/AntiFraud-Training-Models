import os
import json
from sqlalchemy import create_engine, text

def update_training_and_classifier(training_id: str, paths_model: str, path_label: str, acc: str, 
                                   hyperparams:str, per_class:str):
    """
    Fungsi ini mengupdate status training dan melakukan upsert pada tabel classifier_collection
    dengan menggunakan SQLAlchemy untuk menghindari SQL injection dan credential leak.
    """
    # === 1️ Ambil kredensial dari environment variable ===
    DB_USER = os.getenv("DB_USERNAME_LOCAL")              
    DB_PASSWORD = os.getenv("DB_PASSWORD")   
    DB_SERVER = os.getenv("DB_SERVER_LOCAL")          
    DB_NAME = os.getenv("DB_NAME_RESULT", "dbAntiFraud")  
    result = {
    "accuracy": acc,
    "hyperparameters": hyperparams,
    "per_class_metrics": per_class
}

    if not all([DB_USER, DB_PASSWORD, DB_SERVER]):
        raise ValueError(" Environment variable DB_USER, DB_PASSWORD, atau DB_SERVER belum diset!")

    # === 2️ Buat koneksi SQLAlchemy yang aman ===
    connection_string = (
        f"mssql+pyodbc://{DB_USER}:{DB_PASSWORD}@{DB_SERVER}/{DB_NAME}"
        "?driver=ODBC+Driver+17+for+SQL+Server"
    )

    engine = create_engine(connection_string, echo=False, future=True)
    print(" Koneksi database berhasil dibuat dengan aman.")

    # === 4️ Eksekusi query aman ===
    with engine.begin() as conn:  # begin() = auto commit/rollback otomatis
        # Update status training
        conn.execute(
            text("""
                UPDATE data_training_process
                SET training_status = 'SUCCES'
                WHERE id = :training_id
            """),
            {"training_id": str(training_id)}
        )

        # Upsert classifier_collection
        conn.execute(
            text("""
                MERGE classifier_collection AS target
                USING (SELECT :training_id AS training_process_id) AS source
                ON target.training_process_id = source.training_process_id
                WHEN MATCHED THEN
                    UPDATE SET model_path = :paths_model, accuracy = :acc, label_path = :path_label
                WHEN NOT MATCHED THEN
                    INSERT (training_process_id, model_path, accuracy, label_path)
                    VALUES (:training_id, :paths_model, :acc, :path_label);
                    
            """),
            {
                "training_id": training_id,
                "paths_model": paths_model,
                "acc": json.dumps(result, indent=2),
                "path_label": path_label
            }
        )

    print(" Query berhasil dijalankan dan aman dari SQL injection & credential leak.")

if __name__ == "__main__":
    import sys

    training_id = sys.argv[1]
    paths_model = sys.argv[2]
    path_label = sys.argv[3]
    acc = sys.argv[4]
    hyperparams = sys.argv[5]
    per_class = sys.argv[6]

    update_training_and_classifier(training_id, paths_model, path_label, acc, hyperparams, per_class)
