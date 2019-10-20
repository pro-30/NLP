package com.example.st;

import android.content.ActivityNotFoundException;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;
import android.content.Intent;
import android.speech.RecognizerIntent;
import android.widget.TextView;
import android.view.View;

import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_CODE = 100;
    private TextView textOutput;
    FirebaseDatabase fdb = FirebaseDatabase.getInstance();
    final DatabaseReference myRef = fdb.getReference("Text");
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        textOutput= (TextView) findViewById(R.id.textOutput);

        //myRef.setValue("Working");
    }

    public void onClick(View v)

    {
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);

        try {
        startActivityForResult(intent, REQUEST_CODE);
        } catch (ActivityNotFoundException a) {

        }
    }

    @Override


    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        switch (requestCode) {
            case REQUEST_CODE: {
                if (resultCode == RESULT_OK && null != data) {
                    ArrayList<String> result = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
                    textOutput.setText(result.get(0));
                    myRef.setValue(result.get(0));
                }
                break;
            }

        }
    }

}