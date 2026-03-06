# components/payment_handler.py
import razorpay
import hashlib
import hmac
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import streamlit as st
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class PaymentHandler:
    """Handles Razorpay payment integration"""
    
    def __init__(self):
        # Get API keys from environment variables
        self.key_id = os.getenv('RAZORPAY_KEY_ID', '')
        self.key_secret = os.getenv('RAZORPAY_KEY_SECRET', '')
        
        # For testing, use test keys if not set
        if not self.key_id or not self.key_secret:
            self.key_id = 'rzp_test_your_test_key_id'  # Replace with your test key
            self.key_secret = 'your_test_key_secret'    # Replace with your test secret
            logger.warning("Using default test keys. Please set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET in .env")
        
        # Initialize Razorpay client
        self.client = razorpay.Client(auth=(self.key_id, self.key_secret))
        
        # Transaction storage (in production, use database)
        self.transactions_file = Path(__file__).parent.parent / "data" / "transactions.json"
        self._ensure_transactions_file()
    
    def _ensure_transactions_file(self):
        """Ensure transactions file exists"""
        if not self.transactions_file.exists():
            self.transactions_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.transactions_file, 'w') as f:
                json.dump([], f)
    
    def _load_transactions(self) -> list:
        """Load transactions from file"""
        try:
            with open(self.transactions_file, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def _save_transactions(self, transactions: list):
        """Save transactions to file"""
        with open(self.transactions_file, 'w') as f:
            json.dump(transactions, f, indent=2)
    
    def calculate_amount(self, image_count: int = 1, quality: str = "high") -> int:
        """
        Calculate payment amount
        Returns amount in paise (smallest currency unit)
        """
        base_price = 10  # ₹10 per image
        if quality == "high":
            base_price = 50  # ₹50 for high quality
        
        return base_price * image_count * 100  # Convert to paise
    
    def create_payment_order(self, amount: int, currency: str = "INR", 
                            receipt_id: Optional[str] = None, 
                            notes: Optional[Dict] = None) -> Optional[Dict]:
        """
        Create a Razorpay order
        """
        try:
            # Generate receipt ID if not provided
            if not receipt_id:
                receipt_id = f"receipt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create order data
            order_data = {
                "amount": amount,
                "currency": currency,
                "receipt": receipt_id,
                "notes": notes or {}
            }
            
            # Create order
            order = self.client.order.create(data=order_data)
            
            # Store transaction
            transaction = {
                "order_id": order['id'],
                "receipt_id": receipt_id,
                "amount": amount,
                "currency": currency,
                "status": "created",
                "notes": notes,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            transactions = self._load_transactions()
            transactions.append(transaction)
            self._save_transactions(transactions)
            
            logger.info(f"Order created successfully: {order['id']}")
            return order
            
        except Exception as e:
            logger.error(f"Error creating payment order: {e}")
            return None
    
    def verify_payment_signature(self, payment_response: Dict) -> bool:
        """
        Verify payment signature from Razorpay
        """
        try:
            # Get signature components
            order_id = payment_response.get('razorpay_order_id')
            payment_id = payment_response.get('razorpay_payment_id')
            signature = payment_response.get('razorpay_signature')
            
            if not all([order_id, payment_id, signature]):
                logger.error("Missing signature components")
                return False
            
            # Generate signature for verification
            msg = f"{order_id}|{payment_id}"
            generated_signature = hmac.new(
                self.key_secret.encode(),
                msg.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Verify signature
            is_valid = hmac.compare_digest(generated_signature, signature)
            
            if is_valid:
                logger.info(f"Payment signature verified for order: {order_id}")
                # Update transaction status
                self.update_transaction_status(order_id, "success", payment_id)
            else:
                logger.warning(f"Invalid payment signature for order: {order_id}")
                self.update_transaction_status(order_id, "failed", payment_id)
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error verifying payment signature: {e}")
            return False
    
    def update_transaction_status(self, order_id: str, status: str, 
                                 payment_id: Optional[str] = None) -> bool:
        """
        Update transaction status in database
        """
        try:
            transactions = self._load_transactions()
            
            for transaction in transactions:
                if transaction['order_id'] == order_id:
                    transaction['status'] = status
                    if payment_id:
                        transaction['payment_id'] = payment_id
                    transaction['updated_at'] = datetime.now().isoformat()
                    break
            
            self._save_transactions(transactions)
            logger.info(f"Transaction {order_id} updated to status: {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating transaction: {e}")
            return False
    
    def get_transaction(self, order_id: str) -> Optional[Dict]:
        """Get transaction details by order ID"""
        transactions = self._load_transactions()
        for transaction in transactions:
            if transaction['order_id'] == order_id:
                return transaction
        return None
    
    def get_payment_status(self, order_id: str) -> str:
        """Get payment status for an order"""
        transaction = self.get_transaction(order_id)
        return transaction.get('status', 'not_found') if transaction else 'not_found'
    
    def handle_webhook(self, webhook_data: Dict) -> bool:
        """
        Handle Razorpay webhook events
        """
        try:
            event = webhook_data.get('event')
            payload = webhook_data.get('payload', {})
            
            if event == 'payment.captured':
                payment = payload.get('payment', {}).get('entity', {})
                order_id = payment.get('order_id')
                payment_id = payment.get('id')
                
                if order_id:
                    self.update_transaction_status(order_id, "success", payment_id)
                    logger.info(f"Payment captured via webhook: {payment_id}")
                    
            elif event == 'payment.failed':
                payment = payload.get('payment', {}).get('entity', {})
                order_id = payment.get('order_id')
                
                if order_id:
                    self.update_transaction_status(order_id, "failed")
                    logger.warning(f"Payment failed via webhook: {order_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            return False
    
    def is_payment_successful(self, order_id: str) -> bool:
        """Check if payment was successful"""
        status = self.get_payment_status(order_id)
        return status == "success"

# Initialize payment handler
payment_handler = PaymentHandler()