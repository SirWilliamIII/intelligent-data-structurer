"""
Document processor for extracting text from various file types including PDFs and images.
"""

import io
import os
import re
from typing import Tuple, Optional
from pathlib import Path
from loguru import logger

# Import numpy separately since it's used in type hints
try:
    import numpy as np
except ImportError:
    np = None

try:
    import pytesseract
    from PIL import Image
    import cv2
    HAS_OCR = True
except ImportError as e:
    logger.warning(f"OCR dependencies not available: {e}")
    HAS_OCR = False

try:
    import PyPDF2
    import pdfplumber
    HAS_PDF = True
except ImportError as e:
    logger.warning(f"PDF dependencies not available: {e}")
    HAS_PDF = False

class DocumentProcessor:
    """Extract text from various document types."""
    
    def __init__(self):
        self.supported_text_extensions = {'.txt', '.md', '.csv', '.json', '.xml', '.html', '.log'}
        self.supported_pdf_extensions = {'.pdf'}
        self.supported_image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
        
        # OCR configuration
        self.tesseract_config = '--oem 3 --psm 6 -l eng'
        
    def get_file_type(self, filename: str, content_type: Optional[str] = None) -> str:
        """Determine file type from filename and content type."""
        
        file_path = Path(filename.lower())
        extension = file_path.suffix
        
        # Check by extension first
        if extension in self.supported_text_extensions:
            return 'text'
        elif extension in self.supported_pdf_extensions:
            return 'pdf'
        elif extension in self.supported_image_extensions:
            return 'image'
        
        # Check by MIME type if available
        if content_type:
            content_type = content_type.lower()
            if content_type.startswith('text/'):
                return 'text'
            elif content_type == 'application/pdf':
                return 'pdf'
            elif content_type.startswith('image/'):
                return 'image'
        
        # Default to text for unknown types
        return 'text'
    
    async def extract_text(self, file_content: bytes, filename: str, content_type: Optional[str] = None) -> Tuple[str, dict]:
        """Extract text from any supported file type."""
        
        file_type = self.get_file_type(filename, content_type)
        extraction_metadata = {
            'file_type': file_type,
            'original_size': len(file_content),
            'extraction_method': None,
            'ocr_confidence': None,
            'pages_processed': None,
            'preprocessing_applied': []
        }
        
        try:
            if file_type == 'text':
                text, metadata = await self._extract_text_from_text(file_content)
            elif file_type == 'pdf':
                text, metadata = await self._extract_text_from_pdf(file_content, filename)
            elif file_type == 'image':
                text, metadata = await self._extract_text_from_image(file_content, filename)
            else:
                # Fallback to text extraction
                text, metadata = await self._extract_text_from_text(file_content)
            
            extraction_metadata.update(metadata)
            return text, extraction_metadata
            
        except Exception as e:
            logger.error(f"Failed to extract text from {filename}: {e}")
            # Last resort: try as plain text
            try:
                text, metadata = await self._extract_text_from_text(file_content)
                extraction_metadata.update(metadata)
                extraction_metadata['extraction_method'] = 'fallback_text'
                return text, extraction_metadata
            except:
                raise Exception(f"Could not extract text from {filename}: {e}")
    
    async def _extract_text_from_text(self, file_content: bytes) -> Tuple[str, dict]:
        """Extract text from plain text files."""
        
        # Try different encodings
        encodings = ['utf-8', 'utf-16', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                text = file_content.decode(encoding)
                return text, {
                    'extraction_method': 'text_decode',
                    'encoding_used': encoding
                }
            except UnicodeDecodeError:
                continue
        
        raise UnicodeDecodeError("Could not decode file with any supported encoding")
    
    async def _extract_text_from_pdf(self, file_content: bytes, filename: str) -> Tuple[str, dict]:
        """Extract text from PDF files."""
        
        if not HAS_PDF:
            raise ImportError("PDF processing libraries not available. Install PyPDF2 and pdfplumber.")
        
        pdf_file = io.BytesIO(file_content)
        extracted_text = ""
        pages_processed = 0
        extraction_method = None
        
        # Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(pdf_file) as pdf:
                logger.info(f"PDF {filename}: Found {len(pdf.pages)} pages")
                texts = []
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        logger.info(f"PDF {filename} page {page_num}: extracted {len(page_text) if page_text else 0} characters")
                        
                        if page_text and page_text.strip():
                            # Show first 200 chars for debugging
                            preview = page_text[:200].replace('\n', '\\n')
                            logger.info(f"PDF {filename} page {page_num} preview: {preview}")
                            
                            # Filter out obvious binary/corrupted text
                            if self._is_readable_text(page_text):
                                texts.append(page_text)
                                pages_processed += 1
                                logger.info(f"PDF {filename} page {page_num}: ACCEPTED")
                            else:
                                logger.warning(f"PDF {filename} page {page_num}: REJECTED - unreadable text")
                    except Exception as page_error:
                        logger.warning(f"Failed to extract page {page_num} from {filename}: {page_error}")
                
                extracted_text = '\n\n'.join(texts)
                extraction_method = 'pdfplumber'
                logger.info(f"PDF {filename}: Final extracted text length: {len(extracted_text)}")
                
        except Exception as e:
            logger.warning(f"pdfplumber failed for {filename}: {e}, trying PyPDF2")
            
            # Fallback to PyPDF2
            try:
                pdf_file.seek(0)  # Reset file pointer
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                texts = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            # Filter out obvious binary/corrupted text
                            if self._is_readable_text(page_text):
                                texts.append(page_text)
                                pages_processed += 1
                            else:
                                logger.warning(f"Page {page_num} contains unreadable text, skipping")
                    except Exception as page_error:
                        logger.warning(f"Failed to extract page {page_num} from {filename}: {page_error}")
                
                extracted_text = '\n\n'.join(texts)
                extraction_method = 'pypdf2'
                
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed for {filename}: {e2}")
                raise Exception(f"Could not extract text from PDF: {e2}")
        
        # Validate extracted text
        if not extracted_text or not self._is_readable_text(extracted_text):
            raise Exception(f"Could not extract readable text from PDF {filename}")
        
        # Clean up extracted text
        extracted_text = self._clean_pdf_text(extracted_text)
        
        return extracted_text, {
            'extraction_method': extraction_method,
            'pages_processed': pages_processed
        }
    
    async def _extract_text_from_image(self, file_content: bytes, filename: str) -> Tuple[str, dict]:
        """Extract text from image files using OCR."""
        
        if not HAS_OCR:
            raise ImportError("OCR libraries not available. Install pytesseract, Pillow, and opencv-python.")
        
        # Load image
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(file_content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image")
            
            # Preprocess image for better OCR
            preprocessed_image, preprocessing_steps = self._preprocess_image_for_ocr(image)
            
            # Convert to PIL Image for tesseract
            if len(preprocessed_image.shape) == 3:
                # Convert BGR to RGB
                pil_image = Image.fromarray(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(preprocessed_image)
            
            # Perform OCR
            ocr_data = pytesseract.image_to_data(pil_image, config=self.tesseract_config, output_type=pytesseract.Output.DICT)
            
            # Extract text and calculate confidence
            extracted_text = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Clean up OCR text
            cleaned_text = self._clean_ocr_text(extracted_text)
            
            return cleaned_text, {
                'extraction_method': 'tesseract_ocr',
                'ocr_confidence': avg_confidence,
                'preprocessing_applied': preprocessing_steps,
                'words_detected': len([word for word in ocr_data['text'] if word.strip()])
            }
            
        except Exception as e:
            logger.error(f"OCR failed for {filename}: {e}")
            raise Exception(f"Could not extract text from image: {e}")
    
    def _preprocess_image_for_ocr(self, image):
        """Preprocess image to improve OCR accuracy."""
        
        preprocessing_steps = []
        processed_image = image.copy()
        
        # Convert to grayscale
        if len(processed_image.shape) == 3:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            preprocessing_steps.append('grayscale_conversion')
        
        # Resize if image is very small or very large
        height, width = processed_image.shape
        if height < 300 or width < 300:
            # Scale up small images
            scale_factor = max(300 / height, 300 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            processed_image = cv2.resize(processed_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            preprocessing_steps.append(f'upscale_{scale_factor:.1f}x')
        elif height > 3000 or width > 3000:
            # Scale down very large images
            scale_factor = min(3000 / height, 3000 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            processed_image = cv2.resize(processed_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            preprocessing_steps.append(f'downscale_{scale_factor:.1f}x')
        
        # Noise reduction
        processed_image = cv2.medianBlur(processed_image, 3)
        preprocessing_steps.append('noise_reduction')
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        processed_image = clahe.apply(processed_image)
        preprocessing_steps.append('contrast_enhancement')
        
        # Threshold to get binary image
        _, processed_image = cv2.threshold(processed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessing_steps.append('binary_threshold')
        
        return processed_image, preprocessing_steps
    
    def _is_repeating_pattern(self, line: str) -> bool:
        """Check if a line is mostly repeating characters (like dashes, dots, underscores)."""
        
        if len(line) < 10:  # Too short to be a pattern
            return False
        
        # Count occurrences of each character
        char_counts = {}
        for char in line:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # If any single character makes up more than 70% of the line, it's likely a pattern
        max_char_count = max(char_counts.values())
        if max_char_count / len(line) > 0.7:
            return True
        
        # Check for common repeating patterns
        repeating_patterns = ['-', '_', '.', '=', '*', '+', '#', '~', '^']
        for pattern in repeating_patterns:
            if line.count(pattern) > len(line) * 0.6:  # More than 60% of the line
                return True
        
        return False

    def _is_readable_text(self, text: str) -> bool:
        """Check if text is readable (not binary/corrupted)."""
        
        if not text or len(text.strip()) < 10:
            return False
        
        # Calculate ratio of printable characters
        printable_chars = sum(1 for c in text if c.isprintable() or c.isspace())
        total_chars = len(text)
        
        if total_chars == 0:
            return False
        
        printable_ratio = printable_chars / total_chars
        
        # Check for excessive special characters or binary data
        special_chars = sum(1 for c in text if ord(c) > 127)
        special_ratio = special_chars / total_chars
        
        # Text is readable if:
        # - At least 80% printable characters
        # - Less than 30% special/binary characters
        # - Contains some alphanumeric content
        has_alphanumeric = any(c.isalnum() for c in text)
        
        return (printable_ratio >= 0.8 and 
                special_ratio <= 0.3 and 
                has_alphanumeric)

    def _clean_pdf_text(self, text: str) -> str:
        """Aggressively clean text extracted from PDFs."""
        
        # FIRST: Remove null bytes and problematic characters
        text = text.replace('\x00', '')  # Remove null bytes
        text = text.replace('\ufffd', '')  # Remove replacement characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)  # Remove control characters
        
        # Remove page breaks and form feeds
        text = re.sub(r'[\f\x0c]', ' ', text)
        
        # Fix common PDF extraction artifacts
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between words that got merged
        
        # Split into lines and clean aggressively
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Remove excessive spaces within the line
            line = re.sub(r' +', ' ', line)
            
            # Skip very short lines (likely artifacts)
            if len(line) < 3:
                continue
                
            # Skip lines that are just numbers (page numbers)
            if re.match(r'^\d+$', line):
                continue
                
            # Skip lines with mostly special characters
            special_char_ratio = sum(1 for c in line if not c.isalnum() and not c.isspace()) / len(line)
            if special_char_ratio > 0.7:
                continue
            
            # Skip lines that are mostly repeating characters (like dashes, dots, etc.)
            if self._is_repeating_pattern(line):
                continue
            
            cleaned_lines.append(line)
        
        # Join with single newlines
        return '\n'.join(cleaned_lines).strip()
    
    def _clean_ocr_text(self, text: str) -> str:
        """Aggressively clean text extracted from OCR."""
        
        # FIRST: Remove null bytes and problematic characters
        text = text.replace('\x00', '')  # Remove null bytes
        text = text.replace('\ufffd', '')  # Remove replacement characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)  # Remove control characters
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Remove excessive spaces within the line
            line = re.sub(r' +', ' ', line)
            
            # Skip very short lines (likely OCR artifacts)
            if len(line) < 3:
                continue
            
            # Must contain some alphanumeric content
            if not re.search(r'[a-zA-Z0-9]', line):
                continue
            
            # Skip lines with mostly special characters (OCR noise)
            alnum_chars = sum(1 for c in line if c.isalnum())
            if alnum_chars / len(line) < 0.3:  # Less than 30% alphanumeric
                continue
            
            # Skip lines that are mostly repeating characters (like dashes, dots, etc.)
            if self._is_repeating_pattern(line):
                continue
            
            cleaned_lines.append(line)
        
        # Join with single newlines
        return '\n'.join(cleaned_lines).strip()
    
    def get_supported_extensions(self) -> set:
        """Get all supported file extensions."""
        return (self.supported_text_extensions | 
                self.supported_pdf_extensions | 
                self.supported_image_extensions)
    
    def is_supported_file(self, filename: str) -> bool:
        """Check if file type is supported."""
        extension = Path(filename.lower()).suffix
        return extension in self.get_supported_extensions()