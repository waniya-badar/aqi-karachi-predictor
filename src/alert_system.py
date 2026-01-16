"""
Alert System - Sends notifications for hazardous AQI levels
Monitors AQI and sends alerts when thresholds are exceeded
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()


class AlertSystem:
    """Manages AQI alerts and notifications"""
    
    # AQI thresholds for alerts
    THRESHOLDS = {
        'hazardous': 300,
        'very_unhealthy': 200,
        'unhealthy': 150,
        'moderate': 100
    }
    
    def __init__(self):
        """Initialize alert system"""
        self.last_alert_time = {}
        self.alert_cooldown = timedelta(hours=6) 
    
    def check_aqi_level(self, aqi: float) -> tuple:
        """
        Check AQI level and return severity
        
        Args:
            aqi: Current AQI value
        
        Returns:
            (level, should_alert, message)
        """
        if aqi >= self.THRESHOLDS['hazardous']:
            return ('hazardous', True, self._get_hazardous_message(aqi))
        elif aqi >= self.THRESHOLDS['very_unhealthy']:
            return ('very_unhealthy', True, self._get_very_unhealthy_message(aqi))
        elif aqi >= self.THRESHOLDS['unhealthy']:
            return ('unhealthy', True, self._get_unhealthy_message(aqi))
        elif aqi >= self.THRESHOLDS['moderate']:
            return ('moderate', False, self._get_moderate_message(aqi))
        else:
            return ('good', False, self._get_good_message(aqi))
    
    def should_send_alert(self, level: str) -> bool:
        """
        Check if enough time has passed since last alert
        
        Args:
            level: Alert level
        
        Returns:
            bool: True if should send alert
        """
        if level not in self.last_alert_time:
            return True
        
        time_since_last = datetime.now() - self.last_alert_time[level]
        return time_since_last > self.alert_cooldown
    
    def send_alert(self, level: str, aqi: float, message: str):
        """
        Send alert notification
        
        Args:
            level: Alert severity level
            aqi: Current AQI value
            message: Alert message
        """
        # Console alert (always)
        self._console_alert(level, aqi, message)
        
        # Log to file
        self._log_alert(level, aqi, message)
        
        # Update last alert time
        self.last_alert_time[level] = datetime.now()
    
    def _console_alert(self, level: str, aqi: float, message: str):
        """Print alert to console"""
        symbols = {
            'hazardous': '🚨🚨🚨',
            'very_unhealthy': '☠️☠️',
            'unhealthy': '⚠️⚠️',
            'moderate': 'ℹ️',
            'good': '✅'
        }
        
        symbol = symbols.get(level, '⚠️')
        
        print(f"\n{'='*70}")
        print(f"{symbol} AQI ALERT - {level.upper().replace('_', ' ')} {symbol}")
        print(f"{'='*70}")
        print(f"Current AQI: {aqi:.0f}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n{message}")
        print(f"{'='*70}\n")
    
    def _log_alert(self, level: str, aqi: float, message: str):
        """Log alert to file"""
        log_dir = 'data/alerts'
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, 'alert_log.txt')
        
        with open(log_file, 'a') as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"Alert Level: {level.upper().replace('_', ' ')}\n")
            f.write(f"AQI: {aqi:.0f}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Message: {message}\n")
            f.write(f"{'='*70}\n")
    
    def _get_hazardous_message(self, aqi: float) -> str:
        """Get hazardous level message"""
        return f"""
 EMERGENCY - HAZARDOUS AIR QUALITY 

Current AQI: {aqi:.0f} (HAZARDOUS)

IMMEDIATE ACTIONS REQUIRED:
• Stay indoors with windows and doors closed
• Avoid ALL outdoor activities
• Use air purifiers if available
• Wear N95 masks if you must go outside
• Seek medical attention if experiencing breathing difficulties
• Keep emergency contacts ready

VULNERABLE GROUPS AT EXTREME RISK:
• Children and elderly
• People with asthma, heart disease, or lung conditions
• Pregnant women

This is an emergency situation. Follow all health advisories.
        """
    
    def _get_very_unhealthy_message(self, aqi: float) -> str:
        """Get very unhealthy level message"""
        return f"""
 HEALTH ALERT - VERY UNHEALTHY AIR QUALITY

Current AQI: {aqi:.0f} (VERY UNHEALTHY)

RECOMMENDED ACTIONS:
• Everyone should avoid outdoor activities
• Keep windows closed
• Use air conditioning on recirculate mode
• Wear masks (N95 preferred) if outdoor exposure unavoidable
• Monitor health symptoms closely

AFFECTED GROUPS:
• General population at risk of health effects
• Sensitive groups at serious risk

Limit all outdoor exertion. Health effects likely for everyone.
        """
    
    def _get_unhealthy_message(self, aqi: float) -> str:
        """Get unhealthy level message"""
        return f"""
 HEALTH ADVISORY - UNHEALTHY AIR QUALITY

Current AQI: {aqi:.0f} (UNHEALTHY)

RECOMMENDED ACTIONS:
• Sensitive groups should stay indoors
• General population should limit prolonged outdoor activities
• Consider wearing masks during outdoor activities
• Reduce physical exertion outdoors
• Keep rescue inhalers accessible if you have asthma

SENSITIVE GROUPS:
• Children and elderly
• People with heart or lung conditions
• Athletes and outdoor workers

Everyone may begin to experience health effects.
        """
    
    def _get_moderate_message(self, aqi: float) -> str:
        """Get moderate level message"""
        return f"""
ℹ AIR QUALITY MODERATE

Current AQI: {aqi:.0f} (MODERATE)

GUIDANCE:
• Unusually sensitive people should limit prolonged outdoor exertion
• General population can enjoy normal outdoor activities
• Monitor air quality if you have respiratory sensitivities

Air quality is acceptable for most people.
        """
    
    def _get_good_message(self, aqi: float) -> str:
        """Get good level message"""
        return f"""
 AIR QUALITY GOOD

Current AQI: {aqi:.0f} (GOOD)

Air quality is satisfactory. No restrictions on outdoor activities.
Great day to be outside!
        """
    
    def monitor_aqi(self, aqi: float):
        """
        Main monitoring function - checks AQI and sends alerts if needed
        
        Args:
            aqi: Current AQI value
        """
        level, should_alert, message = self.check_aqi_level(aqi)
        
        if should_alert and self.should_send_alert(level):
            self.send_alert(level, aqi, message)
        elif should_alert:
            print(f" Alert suppressed (cooldown period): {level.upper()} - AQI: {aqi:.0f}")
        else:
            print(f" AQI Level: {level.upper()} - {aqi:.0f} (No alert needed)")
    
    def get_alert_summary(self) -> Dict:
        """Get summary of recent alerts"""
        log_file = 'data/alerts/alert_log.txt'
        
        if not os.path.exists(log_file):
            return {'total_alerts': 0, 'last_alert': None}
        
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Count alerts
        total_alerts = content.count('Alert Level:')
        
        # Get last alert time
        lines = content.strip().split('\n')
        last_alert_time = None
        
        for line in reversed(lines):
            if line.startswith('Timestamp:'):
                last_alert_time = line.split('Timestamp:')[1].strip()
                break
        
        return {
            'total_alerts': total_alerts,
            'last_alert': last_alert_time
        }


def run_alert_monitor():
    """Run alert monitoring on current AQI"""
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from src.data_fetcher import AQICNFetcher
    
    print("\n" + "="*70)
    print("AQI ALERT MONITORING SYSTEM")
    print("="*70 + "\n")
    
    # Fetch current AQI
    fetcher = AQICNFetcher()
    data = fetcher.fetch_current_data()
    
    if not data or 'aqi' not in data:
        print(" Failed to fetch current AQI data")
        return
    
    current_aqi = data['aqi']
    
    # Initialize alert system
    alert_system = AlertSystem()
    
    # Monitor AQI
    alert_system.monitor_aqi(current_aqi)
    
    # Show summary
    summary = alert_system.get_alert_summary()
    print(f"\nAlert Summary:")
    print(f"Total alerts logged: {summary['total_alerts']}")
    if summary['last_alert']:
        print(f"Last alert: {summary['last_alert']}")


if __name__ == "__main__":
    run_alert_monitor()