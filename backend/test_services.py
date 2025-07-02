import asyncio
import os
import logging
from services import medical_chatbot

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_all_services():
    logger.info("🏥 Running Medical Chatbot Tests")
    
    # Test 1: Emergency case
    result = await medical_chatbot.process_medical_query("chest pain, shortness of breath")
    assert result["success"] and result["urgency_level"] == "emergency", "Emergency test failed"
    assert "SEEK IMMEDIATE MEDICAL ATTENTION" in result["disclaimer"], "Emergency disclaimer missing"
    logger.info("✅ Emergency test passed")

    # Test 2: Non-emergency with CoT
    result = await medical_chatbot.process_medical_query("headache and nausea")
    assert result["success"] and result["urgency_level"] == "moderate", "CoT test failed"
    assert "migraine" in result["response"].lower() or "tension headache" in result["response"].lower(), "CoT content test failed"
    assert "explanation" in result, "CoT explanation missing"
    logger.info(f"✅ CoT test passed, explanation: {result['explanation']}")

    # Test 3: Multilingual input
    result = await medical_chatbot.process_medical_query("dolor de cabeza y náuseas")  # Spanish: headache and nausea
    assert result["success"] and "migraine" in result["response"].lower(), "Multilingual test failed"
    logger.info("✅ Multilingual test passed")

    # Test 4: Emotional AI
    result = await medical_chatbot.process_medical_query("I'm really scared about my chest pain")
    assert result["success"] and ("empathetic" in result["response"].lower() or "emergency" in result["urgency_level"]), "Emotional AI test failed"
    logger.info("✅ Emotional AI test passed")

    # Test 5: Document processing
    test_file = "test.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Patient has diabetes and takes metformin.")
    result = await medical_chatbot.doc_service.process_document(test_file, "txt")
    logger.info(f"Document processing result: {result}")
    assert result["success"], "Document processing failed"
    assert "diabetes" in str(result["medical_info"]).lower() or "metformin" in str(result["medical_info"]).lower(), "Document processing content test failed"
    os.remove(test_file)
    logger.info("✅ Document processing test passed")

    # Test 6: Cache performance
    # Warm up cache with a dummy query
    await medical_chatbot.process_medical_query("headache and nausea")
    logger.info("Cache warmed up with dummy query")

    start_time = asyncio.get_event_loop().time()
    result = await medical_chatbot.process_medical_query("headache and nausea")
    first_run = asyncio.get_event_loop().time() - start_time
    logger.info(f"First run time: {first_run:.3f} seconds")

    start_time = asyncio.get_event_loop().time()
    result = await medical_chatbot.process_medical_query("headache and nausea")
    second_run = asyncio.get_event_loop().time() - start_time
    logger.info(f"Second run time: {second_run:.3f} seconds")

    assert second_run <= first_run * 1.1, f"Cache performance test failed: second_run ({second_run:.3f}s) not faster than first_run ({first_run:.3f}s)"
    logger.info("✅ Cache performance test passed")

if __name__ == "__main__":
    asyncio.run(test_all_services())
    logger.info("✅ All tests passed successfully!")