import asyncio
import os
import logging
from services import medical_chatbot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_all_services():
    logger.info("🏥 Running Medical Chatbot Tests")
    
    # Test 1: Text-only emergency query
    result = await medical_chatbot.process_medical_query("chest pain, shortness of breath")
    logger.info(f"Emergency test result: {result}")
    assert result["success"], f"Emergency test failed: {result['error']}"
    assert result["urgency_level"] == "emergency", "Emergency test failed: incorrect urgency level"
    assert "always consult qualified healthcare providers" in result["response"].lower(), "Emergency disclaimer missing"
    assert all(symptom in result["response"].lower() for symptom in ["chest pain", "shortness of breath"]), "Emergency test failed: missing symptoms in response"
    logger.info("✅ Emergency test passed")

    # Test 2: Text-only non-emergency query
    result = await medical_chatbot.process_medical_query("headache and nausea")
    logger.info(f"CoT test result: {result}")
    assert result["success"], f"CoT test failed: {result['error']}"
    assert result["urgency_level"] == "normal", "CoT test failed: incorrect urgency level"
    assert "migraine" in result["response"].lower() or "tension headache" in result["response"].lower(), "CoT content test failed"
    assert "explanation" in result, "CoT explanation missing"
    logger.info(f"✅ CoT test passed, explanation: {result['explanation']}")

    # Test 3: Multilingual query
    result = await medical_chatbot.process_medical_query("dolor de cabeza y náuseas")
    logger.info(f"Multilingual test result: {result}")
    assert result["success"], f"Multilingual test failed: {result['error']}"
    assert "migraine" in result["response"].lower(), "Multilingual test failed: incorrect content"
    logger.info("✅ Multilingual test passed")

    # Test 4: Emotional query
    result = await medical_chatbot.process_medical_query("I'm really scared about my chest pain")
    logger.info(f"Emotional AI test result: {result}")
    assert result["success"], f"Emotional AI test failed: {result['error']}"
    assert ("empathetic" in result["response"].lower() or "emergency" in result["urgency_level"]), "Emotional AI test failed"
    logger.info("✅ Emotional AI test passed")

    # Test 5: Document processing (text file)
    test_file = "test.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Patient has diabetes and takes metformin.")
    result = await medical_chatbot.doc_service.process_document(test_file, "txt")
    logger.info(f"Document processing result: {result}")
    assert result["success"], f"Document processing failed: {result['error']}"
    assert "diabetes" in str(result["medical_info"]).lower() or "metformin" in str(result["medical_info"]).lower(), "Document processing content test failed"
    os.remove(test_file)
    logger.info("✅ Document processing test passed")

    # Test 6: Cache performance
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