from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_6wQfHU_F948EotXhGn7UVyYbjjGW2hb6HuQeAsiHeoWAFTBiSN1byr6a3ZJUoR91b2naPX")
index = pc.Index("person-reid")

index.delete(delete_all=True)

print("✅ Database cleared")